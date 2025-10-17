from functools import partial

from torch import nn, Tensor
import torch
from math import floor

import torch.nn.functional as F

from typing import Literal

from torch.utils import checkpoint

from method.nn_util import IdentityMod, ConcatTensors, InvertedConvolution, LayerNorm2d, DynRelPos2d, ApertureAwareAttention, DWConv2d, PatchEmbedIR, PatchUnEmbedIR, \
    ChannelEmbeddingCompression
from method.util import get_activation, get_cnn_attention
from timm.layers import DropPath


class ResidualBlock(nn.Module):
    """

    Args:
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, embed_dim, embed_dim_next, depth, num_heads, heads_range, init_value,
                 ffn_dim=96, layerscale=False, layer_init_values=1e-5,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 resi_connection='1conv', use_pos_map=False):
        super(ResidualBlock, self).__init__()

        self.embed_dim = embed_dim

        self.residual_group = BasicLayer(embed_dim=embed_dim,
                                         depth=depth,
                                         num_heads=num_heads,
                                         heads_range=heads_range,
                                         init_value=init_value,
                                         drop_path=drop_path,
                                         ffn_dim=ffn_dim,
                                         norm_layer=norm_layer,
                                         use_checkpoint=use_checkpoint,
                                         layerscale=layerscale,
                                         layer_init_values=layer_init_values,
                                         )

        embed_dim_conv_in = embed_dim + 2 if use_pos_map else embed_dim
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(embed_dim_conv_in, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(embed_dim_conv_in, embed_dim // 4, 3, 1, 1),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        self.patch_embed = PatchEmbedIR(embed_dim=embed_dim)

        self.patch_unembed = PatchUnEmbedIR(embed_dim=embed_dim)

        self.final_op = ChannelEmbeddingCompression(embed_dim, embed_dim_next) if (
                embed_dim != embed_dim_next) else IdentityMod()

        self.use_pos_map = use_pos_map

    def forward(self, x, x_size, pos_map=None, att_range_factor=None, **kwargs):

        rb_stl_out = self.residual_group(x, att_range_factor)

        rb_unembed = self.patch_unembed(rb_stl_out, x_size)

        rb_unembed = torch.cat((rb_unembed, pos_map), dim=1) if self.use_pos_map else rb_unembed
        rb_last_conv = self.conv(rb_unembed)

        rb_re_enbed = self.patch_embed(rb_last_conv)

        rb_res = rb_re_enbed + x

        return self.final_op(rb_res, x_size)

class BasicLayer(nn.Module):
    def __init__(self, embed_dim, depth, num_heads,
                 init_value: float, heads_range: float,
                 ffn_dim=96, drop_path=0., norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.Relpos = DynRelPos2d(embed_dim, num_heads, init_value, heads_range)

        # build blocks
        self.blocks = nn.ModuleList([
            ApertureAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim,
                                   layerscale=layerscale, norm_layer=norm_layer, layer_init_values=layer_init_values,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

    def forward(self, x, att_range_factor=None):
        b, h, w, d = x.size()


        rel_pos = self.Relpos((h, w), range_factor=att_range_factor)

        for blk in self.blocks:
            if self.use_checkpoint:
                tmp_blk = partial(blk, attention_rel_pos=rel_pos)
                x = checkpoint.checkpoint(tmp_blk, x)
            else:
                x = blk(x, attention_rel_pos=rel_pos)

        return x

class ApertureAttentionBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False,
                 norm_layer=nn.LayerNorm, layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.attention_layer_norm = norm_layer(self.embed_dim, eps=1e-6)
        self.attention = ApertureAwareAttention(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = norm_layer(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def forward(self, x: torch.Tensor, attention_rel_pos=None):
        x = x + self.pos(x) # InitiaL 3 X 3 dwconv
        if self.layerscale:
            x = x + self.drop_path(
                self.gamma_1 * self.attention(self.attention_layer_norm(x), attention_rel_pos))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(
                self.attention(self.attention_layer_norm(x), attention_rel_pos))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x

class BlockMod(nn.Module):
    def __init__(self, channels: int, dw_expand: float = 1., ffn_expand: int = 2, drop_out_rate: float = 0.,
                 attention_type: Literal['CA', 'SCA'] = 'CA',
                 activation_type: Literal['GELU', 'SG', 'Identity', 'ReLU', 'LReLU', 'Tanh', 'Sigmoid'] = 'GELU',
                 inverted_conv: bool = True, kernel_size: int = 3,
                 use_pos_map: bool = False, depth: int = None):
        """
        Modified block from NAFNet with optional CoordConv on th first convolution.
        Args:
            channels: Number of input channels
            dw_expand: Expansion factor for sub block 1
            ffn_expand: Expansion factor for sub block 2
            drop_out_rate: Dropout rate
            attention_type: Type of attention to use
            activation_type: Type of activation to use
            inverted_conv: Whether to use inverted convolution
            kernel_size: size of all convolution kernels
            use_pos_map: Whether coordconv is used
            depth: Depth of the block in the network, used for positional map scaling
        """
        super().__init__()
        dw_channel = int(channels * dw_expand) if int(channels * dw_expand) % 2 == 0 else int(channels * dw_expand) + 1

        self.cat = ConcatTensors() if use_pos_map else IdentityMod()

        if inverted_conv:
            self.conv1 = InvertedConvolution(in_channels=channels + 2 if use_pos_map else channels,
                                             out_channels=dw_channel,
                                             kernel_size=kernel_size, padding='same', bias=True)
        else:
            self.conv1 = nn.Conv2d(in_channels=channels + 2 if use_pos_map else channels, out_channels=dw_channel,
                                   kernel_size=kernel_size, padding='same', stride=1, bias=True)

        # Activation
        self.activation = get_activation(activation_type)

        # Channel Attention
        self.attention = get_cnn_attention(attention_type)(
            dw_channel // 2 if activation_type == 'SG' else dw_channel)

        self.conv2 = nn.Conv2d(in_channels=dw_channel // 2 if activation_type == 'SG' else dw_channel,
                               out_channels=channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)


        # sub block 1 done
        ffn_channel = floor(ffn_expand * channels)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # second activation call
        self.conv4 = nn.Conv2d(in_channels=ffn_channel // 2 if activation_type == 'SG' else ffn_channel,
                               out_channels=channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # sub block 2 done

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

        # save use_pos_map and depth metadata for forward
        self.use_pos_map = use_pos_map
        self.depth = depth

    def forward(self, source: Tensor, pos_map: Tensor = None) -> Tensor:
        x = source

        x = self.norm1(x)

        x = self.cat(x, pos_map)
        x = self.conv1(x)
        x = self.activation(x)

        att = self.attention(x)

        x = x * att
        x = self.conv2(x)

        x = self.dropout1(x)

        y = source + x * self.beta

        x = self.conv3(self.norm2(y))
        x = self.activation(x)
        x = self.conv4(x)

        x = self.dropout2(x)

        ret = y + x * self.gamma

        return ret

class FeedForwardNetwork(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            activation_fn=F.gelu,
            dropout=0.0,
            activation_dropout=0.0,
            layernorm_eps=1e-6,
            subln=False,
            subconv=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        """
        x: (b h w c)
        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x