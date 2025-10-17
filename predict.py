from torch import load, no_grad
from torchvision.transforms.functional import to_pil_image

from method.model import Bokehlicious
from method.config import bokehlicious_size_builder

from dataset.util import load_image
from util.parser import get_predict_parser

if __name__ == "__main__":
    parser = get_predict_parser()
    args = parser.parse_args()

    config = bokehlicious_size_builder(args.size)

    model = Bokehlicious(**config)

    print(f"Initialized {args.size} Bokehlicious model on {args.device}")

    checkpoint = f'./checkpoints/{args.size}.pt'

    state_dict = load(checkpoint)

    model.load_state_dict(state_dict)

    model.to(args.device)
    model.eval()

    print(f"Loaded weights from {checkpoint}")

    net_input = load_image(args.img_path, target_av=args.av, max_dim=args.max_dim, device=args.device)

    with no_grad():
        out = model(**net_input)

    print(f"Rendered {args.img_path} at f{args.av}")

    out_img = to_pil_image(out.cpu().detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy())

    output_file = f'{args.out_path}/net_{args.size}_f{args.av}_{args.img_path.split("/")[-1]}'
    out_img.save(output_file)

    print(f"Saved result to {output_file}")