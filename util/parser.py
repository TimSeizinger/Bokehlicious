from argparse import ArgumentParser

def get_base_parser():
    parser = ArgumentParser()
    parser.add_argument('-out_path', type=str, default='./output')
    parser.add_argument('-image_format', type=str, default='jpg')
    return parser

def add_network_args(parser: ArgumentParser):
    parser.add_argument('-size', type=str, required=True, choices=['small', 'large'])
    parser.add_argument('-device', type=str, default='cuda', choices=['cuda', 'cpu'])

def add_img_args(parser: ArgumentParser):
    parser.add_argument('-img_path', type=str, required=True)
    parser.add_argument('-av', type=float, default=2.0)
    parser.add_argument('-max_dim', type=int, default=2000)
    parser.add_argument('-min_divisor', type=int, default=4)

def get_predict_parser():
    parser = get_base_parser()
    add_network_args(parser)
    add_img_args(parser)
    return parser

def get_eval_parser():
    parser = get_base_parser()
    add_network_args(parser)
    parser.add_argument('-dataset', type=str, required=True, choices=['RealBokeh', 'RealBokeh_bin', 'EBB400', 'EBB_Val294'])
    parser.add_argument('--save_outputs', action='store_true')
    return parser