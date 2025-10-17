import torch
from torch.nn import Module, LayerNorm
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List

from torch.nn.init import trunc_normal_

from method.blocks import BlockMod, ResidualBlock
from method.nn_util import PatchEmbedIR, PatchUnEmbedIR, IdentityMod, SkipConnection

class Bokehlicious(Module):

    def __init__(self,
                 # General Args
                 in_chans=3, img_range=1., dynamic_conv_k=4,
                 # In Stage Args
                 in_stage_use_pos_map: bool = True,
                 in_stage_use_bokeh_strength_map: bool = False,
                 # UNet Args
                 u_width=32,
                 u_depth=2,
                 u_block_config=None,
                 u_skip_connections=None,
                 # Enc Args
                 enc_blk_nums: List[int] = None,
                 enc_blks_use_pos_map: List[bool] = None,
                 # Deep Feature Extractor args
                 embed_dims=None,
                 depths=None,
                 num_heads=None,
                 init_values=None,
                 heads_ranges=None,
                 mlp_ratios=None,
                 drop_path_rate=0.1,
                 norm_layer=LayerNorm,
                 patch_norm=True,
                 use_checkpoints=None,
                 chunkwise_recurrents=None,
                 layerscales=None,
                 layer_init_values=1e-6,
                 positional_dfe=False,
                 use_dfe_norm_layer=True,
                 positional_conv_last=False,
                 # Dec Args
                 dec_blk_nums: List[int] = None,
                 dec_blks_use_pos_map: List[bool] = None,
                 # Out Stage Args
                 out_stage_use_pos_map: bool = True
                 ):
        super().__init__()

        self.in_chans = in_chans
        self.img_range = img_range

        # In Stage Args
        self.in_stage_use_pos_map = in_stage_use_pos_map
        self.in_stage_use_bokeh_strength_map = in_stage_use_bokeh_strength_map

        self.u_width = u_width
        self.u_depth = u_depth
        self.u_block_config = u_block_config or {'dw_expand': 1., 'ffn_expand': 2.,
                                                 'drop_out_rate': 0, 'attention_type': 'CA',
                                                 'activation_type': 'GELU', 'kernel_size': 3,
                                                 'inverted_conv': True,
                                                 }

        self.u_skip_connections = u_skip_connections or [True for _ in range(u_depth)]

        self.enc_blk_nums = enc_blk_nums or [1 for _ in range(u_depth)]
        self.enc_blks_use_pos_map = enc_blks_use_pos_map or [True for _ in range(u_depth)]


        # Deep Feature Extractor args
        self.num_blocks = len(embed_dims)
        self.embed_dims = embed_dims or [192 for _ in range(self.num_blocks)]
        self.depths = depths or [6 for _ in range(self.num_blocks)]
        self.num_heads = num_heads or [6 for _ in range(self.num_blocks)]
        self.init_values = init_values or [2 for _ in range(self.num_blocks)]
        self.heads_ranges = heads_ranges or [6 for _ in range(self.num_blocks)]
        self.mlp_ratios = mlp_ratios or [2 for _ in range(self.num_blocks)]
        self.drop_path_rate = drop_path_rate
        self.chunkwise_recurrents = chunkwise_recurrents or [True for _ in range(self.num_blocks)]
        self.layerscales = layerscales or [False for _ in range(self.num_blocks)]
        self.use_checkpoints = use_checkpoints or [False for _ in range(self.num_blocks)]
        self.positional_dfe = positional_dfe
        self.layer_init_values = layer_init_values
        self.patch_norm = patch_norm
        self.norm_layer = norm_layer
        self.use_dfe_norm_layer = use_dfe_norm_layer
        self.positional_conv_last = positional_conv_last

        # Dec Args
        self.dec_blk_nums = dec_blk_nums or [1 for _ in range(u_depth)]
        self.dec_blks_use_pos_map = dec_blks_use_pos_map or [True for _ in range(u_depth)]

        # Out Stage Args
        self.out_stage_use_pos_map = out_stage_use_pos_map

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################

        extra_in_channels = 0
        extra_in_channels += 2 if self.in_stage_use_pos_map else 0
        extra_in_channels += 1 if self.in_stage_use_bokeh_strength_map else 0

        self.in_stage = nn.Conv2d(in_channels=self.in_chans + extra_in_channels, out_channels=self.u_width,
                      kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.in_stage_2 = nn.Conv2d(in_channels=self.u_width, out_channels=self.u_width,
                      kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = self.u_width
        u_depths = range(0, self.u_depth)
        for num, depth, use_pos_map in \
                zip(self.enc_blk_nums, u_depths, self.enc_blks_use_pos_map):
            self.encoders.append(
                nn.ModuleList(
                    [BlockMod(chan, **self.u_block_config, depth=depth, use_pos_map=use_pos_map)
                     for _ in range(num)]
                )
            )
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan = chan * 2

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.conv_prep = nn.Conv2d(chan, self.embed_dims[0], 3, 1, 1)
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)


        self.patch_embed = PatchEmbedIR(embed_dim=self.embed_dims[0],
                                            norm_layer=self.norm_layer if self.patch_norm else None)

        self.patch_unembed = PatchUnEmbedIR(embed_dim=self.embed_dims[-1])

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        # build layers
        self.blocks = nn.ModuleList()
        for i_block in range(self.num_blocks):
            self.blocks.append(ResidualBlock(
                embed_dim=self.embed_dims[i_block],
                embed_dim_next=
                self.embed_dims[i_block + 1] if i_block < self.num_blocks - 1 else self.embed_dims[i_block],
                depth=self.depths[i_block],
                num_heads=self.num_heads[i_block],
                init_value=self.init_values[i_block],
                heads_range=self.heads_ranges[i_block],
                ffn_dim=int(self.mlp_ratios[i_block] * self.embed_dims[i_block]),
                drop_path=dpr[sum(self.depths[:i_block]):sum(self.depths[:i_block + 1])],
                norm_layer=self.norm_layer,
                use_checkpoint=self.use_checkpoints[i_block],
                layerscale=self.layerscales[i_block],
                layer_init_values=self.layer_init_values,
                use_pos_map=self.positional_dfe,
            ))

        self.norm = nn.LayerNorm(self.embed_dims[-1], eps=1e-6) if self.use_dfe_norm_layer else IdentityMod()

        self.conv_after_body = nn.Conv2d(self.embed_dims[-1], self.embed_dims[0], 3, 1, 1)

        #####################################################################################################
        ################################ 3, bokeh image reconstruction ######################################
        conv_last_extra_channels = 0
        conv_last_extra_channels += 2 if self.positional_conv_last else 0
        self.conv_last = nn.Conv2d(self.embed_dims[0] + conv_last_extra_channels, chan, 3, 1, 1)

        self.decoders = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.ups = nn.ModuleList()

        for num, use_pos_map, skip_connection, depth in \
                zip(self.dec_blk_nums, self.dec_blks_use_pos_map,
                    self.u_skip_connections, u_depths.__reversed__()):
            self.ups.append(
                nn.Sequential(
                    # Expand channels
                    nn.Conv2d(in_channels=chan, out_channels=2 * chan, kernel_size=1, bias=False),
                    # Upsample
                    nn.PixelShuffle(2))
                )
            self.skips.append(SkipConnection() if skip_connection else IdentityMod())
            chan = chan // 2
            self.decoders.append(
                nn.ModuleList(
                    [BlockMod(chan, **self.u_block_config,
                              use_pos_map=use_pos_map, depth=depth)
                     for x in range(num)]
                )
            )

        extra_out_channels = 0
        extra_out_channels += 2 if self.out_stage_use_pos_map else 0
        self.out_stage = nn.Conv2d(in_channels=u_width + extra_out_channels, out_channels=in_chans,
                      kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x, pos_map=None, bokeh_strength=None):
        x_size = (x.shape[2], x.shape[3])

        # print(f"  X shape before patch_embed: {x.shape}")

        x = self.patch_embed(x)

        for block_num, block in enumerate(self.blocks):
            x = block(x, x_size, pos_map=pos_map, att_range_factor=bokeh_strength)

        # print(f"  X shape after layers: {x.shape}")
        x = self.norm(x)  # can be deactivated to a IdentityMod() if necessary via self.dfe_norm_layer = False

        x = self.patch_unembed(x, x_size)
        # print(f"  X shape after unembed: {x.shape}")
        return x

    def forward(self, source: Tensor, bokeh_strength: Tensor = None, pos_map: Tensor = None,
                bokeh_strength_map: Tensor = None, **kwargs):

        self.mean = self.mean.type_as(source)
        source_mean = (source - self.mean) * self.img_range

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################

        x = torch.cat((source_mean, pos_map), dim=1) if self.in_stage_use_pos_map else source
        x = torch.cat((x, bokeh_strength_map), dim=1) if self.in_stage_use_bokeh_strength_map else x

        x = self.in_stage(x)
        x = self.in_stage_2(x)  # Having two convolutions as in stage has better results

        # print(f"X shape after in_stage: {x.shape}")

        encs = []
        for encoder, down, use_pos_map, depth in (
                zip(self.encoders, self.downs, self.enc_blks_use_pos_map, range(0, self.u_depth))):
            if pos_map is not None:
                pos_map_e = F.interpolate(pos_map, scale_factor=1 / 2 ** depth, mode='bilinear', align_corners=False)
            else:
                pos_map_e = None
            for blk in encoder:
                x = blk(x, pos_map=pos_map_e)
            encs.append(x)
            x = down(x)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################

        # print(f"X shape before Deep Feature Extraction: {x.shape} <---------------------- Deep Feature Extractor Start")
        x_prep = self.conv_prep(x)
        # print(f" X shape before RRTB blocks: {x_prep.shape}")
        pos_map_t = F.interpolate(pos_map, scale_factor=1 / 2 ** self.u_depth, mode='bilinear', align_corners=False) \
            if self.positional_dfe or self.positional_conv_last else None
        x_after_body = self.forward_features(x_prep, bokeh_strength=bokeh_strength, pos_map=pos_map_t)
        # print(f" X shape after RRTB blocks: {x_after_body.shape}")
        res = self.conv_after_body(x_after_body) + x_prep

        #res = self.dfe_apply_weights2(res, bokeh_strength)

        #####################################################################################################
        ################################ 3, bokeh image reconstruction ######################################

        res = torch.cat((res, pos_map_t), dim=1) if self.positional_conv_last else res
        x = x + self.conv_last(res)

        # print(f"X shape after conv_last: {x.shape}  <-------------------------------------- Deep Feature Extractor End")

        for decoder, up, skip, enc_skip, use_pos_map_d, depth in (
                zip(self.decoders, self.ups, self.skips, encs[::-1],
                    self.dec_blks_use_pos_map, range(0, self.u_depth).__reversed__())):
            pos_map_d = F.interpolate(pos_map, scale_factor=1 / 2 ** depth, mode='bilinear', align_corners=False) \
                if use_pos_map_d else None
            x = up(x)
            x = skip(x, enc_skip)
            for blk in decoder:
                x = blk(x, pos_map=pos_map_d)

        x = torch.cat((x, pos_map), dim=1) if self.out_stage_use_pos_map else x
        x = self.out_stage(x)

        x = x / self.img_range + self.mean

        return x + source