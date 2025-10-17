def bokehlicious_size_builder(size: str):
    if 'small' in size:
        small_config = {
            'u_width': 16,
            'num_embed_dims': 96,
            'num_heads': 3,
            'dfe_depth': 3,
            'dfe_blocks': 3,
            'dfe_mlp_ratio': 2,
        }
        if size == "small_bin":
            return bokehlicious_config_builder(strength_enc=False, **small_config, )
        elif size == "small":
            return bokehlicious_config_builder(**small_config)
    elif 'large' in size:
        if size == "large_bin":
            return bokehlicious_config_builder(strength_enc=False)
        elif size == "large":
            return bokehlicious_config_builder()
    raise ValueError(f'Unknown size "{size}"')



def bokehlicious_config_builder(
        in_chans = 3,
        im_range = 1.,

        positional_enc = True,
        positional_dfe = True,
        positional_dec = True,
        strength_enc = True,

        u_width=32,
        dfe_depth = 6,
        dfe_blocks = 6,
        num_embed_dims = 192,
        num_heads = 6,
        init_head_range = 2,
        dyn_head_range = 9,
        dfe_mlp_ratio = 2,
        u_depth = 2,
):

    general_args = {'in_chans': in_chans,
                    'img_range': im_range,
                    }

    in_stage_args = {'in_stage_use_pos_map': positional_enc,
                     'in_stage_use_bokeh_strength_map': strength_enc,
                     }

    u_block_config = {'dw_expand': 1., 'ffn_expand': 2.,
                      'drop_out_rate': 0, 'attention_type': 'CA',
                      'activation_type': 'GELU', 'kernel_size': 3,
                      'inverted_conv': True,
                      }

    u_net_args = {'u_width': u_width,
                  'u_depth': u_depth,
                  'u_block_config': u_block_config,
                  'u_skip_connections': [True for _ in range(u_depth)],
                  }

    encoder_args = {'enc_blk_nums': [1 for _ in range(u_net_args['u_depth'])],
                    'enc_blks_use_pos_map': [positional_enc for _ in range(u_net_args['u_depth'])],
                    }

    deep_feature_extractor_args = {'embed_dims': [num_embed_dims for _ in range(dfe_depth)],
                                   'depths': [dfe_blocks for _ in range(dfe_depth)],
                                   'num_heads': [num_heads for _ in range(dfe_depth)],
                                   'init_values': [init_head_range for _ in range(dfe_depth)],
                                   'heads_ranges': [dyn_head_range for _ in range(dfe_depth)],
                                   'mlp_ratios': [dfe_mlp_ratio for _ in range(dfe_depth)],
                                   'drop_path_rate': 0.10,
                                   'chunkwise_recurrents': [True for _ in range(dfe_depth)],
                                   'layerscales': [False for _ in range(dfe_depth)],
                                   'use_checkpoints': [False for _ in range(dfe_depth)],
                                   'positional_dfe': positional_dfe,
                                   'patch_norm': True,
                                   'use_dfe_norm_layer': True,
                                   'positional_conv_last': positional_dfe,
                                   }

    decoder_args = {'dec_blk_nums': ([2 for _ in range(u_net_args['u_depth'] - 1)] + [4]),
                    'dec_blks_use_pos_map': [positional_dec for _ in range(u_net_args['u_depth'])],
                    }

    out_stage_args = {'out_stage_use_pos_map': positional_dec,
                      }

    bokehlicious_config = {
        **general_args,
        **in_stage_args,
        **u_net_args,
        **encoder_args,
        **deep_feature_extractor_args,
        **decoder_args,
        **out_stage_args,
    }

    return bokehlicious_config