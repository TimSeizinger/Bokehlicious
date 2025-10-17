import warnings
from typing import Tuple, Optional

import torch
from timm.layers import to_2tuple
from torch import nn, Tensor, cat, sqrt, ones, zeros
from torch.autograd import Function

from torch.nn.init import trunc_normal_


class IdentityMod(nn.Module):
    def __init__(self):
        super(IdentityMod, self).__init__()

    def forward(self, x=None, *args, **kwargs) -> Tensor:
        return x

class ConcatTensors(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return cat((x, y), dim=1)

class SkipConnection(nn.Module):
    def __init__(self,):
        super(SkipConnection, self).__init__()

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return x + skip

class ApplyVectorWeights(nn.Module):
    def __init__(self):
        super(ApplyVectorWeights, self).__init__()

    def forward(self, x: Tensor, weights: Tensor) -> Tensor:
        return x * weights

class ChannelEmbeddingCompression(nn.Module):
    def __init__(self, embed_dim, embed_dim_next):
        super().__init__()
        self.patch_unembed = PatchUnEmbedIR(embed_dim=embed_dim)
        self.conv = nn.Conv2d(embed_dim, embed_dim_next, 1, 1, 0)
        self.patch_embed = PatchEmbedIR(embed_dim=embed_dim_next)

    def forward(self, x, x_size):
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x)
        return x

class InvertedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=bias)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, stride=1, groups=out_channels, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """
        x: (b h w c)
        """
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x

class ChannelAttention(nn.Module):
    """
    Channel attention module with optional attention weights.
    """
    def __init__(self, num_channel: int):
        """

        :param num_channel: Number of channels in the input tensor
        :param apply_att_weights: Should attention weights be applied in the forward pass?
        """
        super(ChannelAttention, self).__init__()

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=num_channel, out_channels=num_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channel // 2, out_channels=num_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, num_channel, apply_att_weights=False):
        """

        :param num_channel: Number of channels in the input tensor
        :param apply_att_weights: Should attention weights be applied in the forward pass?
        """
        super(SimplifiedChannelAttention, self).__init__()

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.apply1 = ApplyVectorWeights() if apply_att_weights else IdentityMod()

    def forward(self, x: Tensor, att_weights: Tensor = None) -> Tensor:
        x = self.model(x)
        return self.apply1(x=x, weights=att_weights)

class ApertureAwareAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads # key dim is channel size of the key matrix for each head
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2) # LEPE is Local Context Enhancement in the paper

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos):
        bsz, h, w, _ = x.size()

        mask_h, mask_w = rel_pos


        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)

        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)

        mask_w = mask_w.unsqueeze(1).expand(-1, h, -1, -1, -1)


        qk_mat_w = qk_mat_w + mask_w

        qk_mat_w = torch.softmax(qk_mat_w, -1)

        v = qk_mat_w @ v

        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)

        mask_h = mask_h.unsqueeze(1).expand(-1, w, -1, -1, -1)

        qk_mat_h = qk_mat_h + mask_h
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = qk_mat_h @ v

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNormFunction(Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.channels = channels
        self.register_parameter('weight', nn.Parameter(ones(channels)))
        self.register_parameter('bias', nn.Parameter(zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class DynRelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        """
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        """
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        self.register_buffer('angle', angle)

    def generate_1d_decay(self, l: int, range_factor: Tensor):
        """
        generate 1d decay mask, the result is l*l
        """

        range_factor=abs(range_factor)

        bs = range_factor.size(0)
        ###print(f"Generating 1D decay mask for batch size: {bs} and length: {l}")

        heads_ranges = self.heads_range * torch.arange(self.num_heads, dtype=torch.float) / self.num_heads
        heads_ranges = heads_ranges.to(range_factor.device)
        ###print("Heads Ranges:")
        ###print(heads_ranges)
        ###print(f"Heads Ranges shape: {heads_ranges.shape}")

        range_factor = torch.sqrt(torch.sqrt(range_factor))  # (n) # give extra weight to smaller values
        range_factor = range_factor[:, None]  # (n 1)
        ###print("Range Factor:")
        ###print(range_factor)
        ###print(f"Range Factor shape: {range_factor.shape}")

        ranges = (-self.initial_value - heads_ranges.repeat(bs, 1) * range_factor)
        ###print("Ranges:")
        decay = torch.log(1 - 2 ** ranges)  # (b n)
        ###print("Decay:")
        ###print(decay)
        ###print(f"Decay shape: {decay.shape}")

        index = torch.arange(l).to(decay)
        ###print("Index:")
        ###print(index)
        ###print(f"Index shape: {index.shape}")
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        # extend mask to batch size with one channel for each
        mask = mask[None, None, :, :]  # (1 1 l l)
        ###print("Mask before decay application:")
        ###print(mask)
        ###print(f"Mask shape : {mask.shape}")

        mask = mask * decay[:, :, None, None]  # (b n l l)
        ###print("Mask after decay application:")
        ###print(mask)
        ###print(f"Mask shape: {mask.shape}")

        return mask

    def forward(self, slen: Tuple[int], range_factor: Tensor):
        mask_h = self.generate_1d_decay(slen[0], range_factor=range_factor)  # x axis decay
        mask_w = self.generate_1d_decay(slen[1], range_factor=range_factor)  # y axis decay
        return mask_h, mask_w

class PatchEmbedIR(nn.Module):
    r""" Image to Patch Embedding

    Args:
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self, embed_dim=96, norm_layer=None):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (b c h w) -> (b h w c)
        if self.norm is not None:
            # print("Using norm layer")
            x = self.norm(x)
        return x


class PatchUnEmbedIR(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        embed_dim (int): Number of linear projection output channels.
    """

    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.view(B, self.embed_dim, x_size[0], x_size[1]) # B Ph*Pw C
        return x