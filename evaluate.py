from os import makedirs

from torch import load, no_grad, clamp, Tensor
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from dataset.loader import RealBokeh, EBB
from dataset.util import Mode
from method.config import bokehlicious_size_builder
from method.model import Bokehlicious
from util.parser import get_eval_parser

from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity as lpips

def append_av(av_dict, key, value):
    if key in av_dict:
        av_dict[key].append(value)
    else:
        av_dict[key] = [value]

def preprocess_batch(batch):
    for k, v in batch.items():
        if isinstance(v, Tensor):
            batch[k] = v.unsqueeze(0).cuda()
    return batch

if __name__ == "__main__":
    parser = get_eval_parser()
    args = parser.parse_args()

    config = bokehlicious_size_builder(f"{args.size}{'_bin' if (args.dataset == 'RealBokeh_bin') else ''}")

    model = Bokehlicious(**config)

    print(f"Initialized {args.size} Bokehlicious model on {args.device}")

    checkpoint = f"./checkpoints/{args.size}{'_bin' if args.dataset == 'RealBokeh_bin' else '' if args.dataset == 'RealBokeh' else f'_{args.dataset}'}.pt"

    state_dict = load(checkpoint)

    model.load_state_dict(state_dict)

    model.to(args.device)
    model.eval()

    print(f"Loaded weights from {checkpoint}")

    if args.dataset == "RealBokeh":
        dataloader = RealBokeh(data_path="./dataset/RealBokeh_3MP", mode=Mode.TEST, device=args.device)
    elif args.dataset == "RealBokeh_bin":
        dataloader = RealBokeh(data_path="./dataset/RealBokeh_3MP", mode=Mode.TEST, binary_bokeh=True, device=args.device)
    elif args.dataset == "EBB400":
        dataloader = EBB(data_path="./dataset/EBB400", mode=Mode.VAL, device=args.device)
    elif args.dataset == "EBB_Val294":
        dataloader = EBB(data_path="./dataset/EBB_Val294", mode=Mode.VAL, device=args.device)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Initialized {args.dataset} dataloader")

    print(f"Calculating metrics for RealBokeh {args.size} on {args.dataset} dataset...")

    if args.save_outputs:
        makedirs(f"{args.out_path}/{args.dataset}/", exist_ok=True)
        print(f"Saving outputs to \"{args.out_path}/{args.dataset}/\"!")
    else:
        print("Not saving outputs, include --save_output flag to save outputs!")

    lpips_vals = []
    ssim_vals = []
    psnr_vals = []

    lpips_avs = {}
    ssim_avs = {}
    psnr_avs = {}

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        with no_grad():
            output = clamp(model(**preprocess_batch(batch)), 0, 1)

            psnr_val = psnr(output, batch['target'], data_range=1.0).item()
            ssim_val = ssim(output, batch['target'], data_range=1.0).item()
            lpips_val = lpips(output, batch['target'], normalize=True).item()

        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)
        lpips_vals.append(lpips_val)

        if args.dataset == "RealBokeh":
            av = batch['image_name'].split("_")[1]
            append_av(lpips_avs, av, lpips_val)
            append_av(ssim_avs, av, ssim_val)
            append_av(psnr_avs, av, psnr_val)

        to_pil_image(output.squeeze(0).cpu()).save(f"{args.out_path}/{args.dataset}/{batch['image_name']}.{args.image_format}")

    print(f"Results for Bokehlicious {args.size} on {args.dataset}")
    print(f"Mean PSNR: {sum(psnr_vals) / len(psnr_vals):.3f}")
    print(f"Mean SSIM: {sum(ssim_vals) / len(ssim_vals):.4f}")
    print(f"Mean LPIPS: {sum(lpips_vals) / len(lpips_vals):.4f}")

    if args.dataset == "RealBokeh":
        print("------------------------------------------------------")
        for key in sorted(lpips_avs, key=lambda x: float(x.split("f")[-1])):
            print(f"Mean PSNR {key}: {sum(psnr_avs[key]) / len(psnr_avs[key]):.4f}")
            print(f"Mean SSIM {key}: {sum(ssim_avs[key]) / len(ssim_avs[key]):.4f}")
            print(f"Mean LPIPS {key}: {sum(lpips_avs[key]) / len(lpips_avs[key]):.3f}")
            print("------------------------------------------------------")
