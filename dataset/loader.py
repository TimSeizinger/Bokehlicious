import cv2
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from json import load
from typing import Union

from dataset.util import Mode, calculate_aperture_embedding, generate_maps, build_input_dict, crop_to_divisible


class RealBokeh(Dataset):
    """
    Dataset class for the RealBokeh dataset.
        :param data_path: Path to the dataset directory with train/val/test subdirectories. Only test is needed for testing!
        :param mode: one of the modes defined in the Mode enum.
        :param binary_bokeh: whether to use binary bokeh strength of F22.0 and F2.0, RealBokeh_bin in the paper.
        :param defocus_deblur_mode: If true, reverses input and outputs, RealDefocus in the paper
    """
    def __init__(self, data_path: Union[str, Path], mode: Mode,
                 binary_bokeh: bool = False,
                 defocus_deblur_mode: bool = False,
                 device: str = 'cuda'
                 ):
        self._data_path: Path = Path(data_path) if isinstance(data_path, str) else data_path
        assert self._data_path.exists(), f"Data directory {self._data_path} does not exist!"

        self.defocus_deblur_mode = defocus_deblur_mode
        if self.defocus_deblur_mode:
            print("Dataset is in Defocus Deblur mode!")

        self._mode = mode
        # Iterate over individual samples of each scene for validation and test, over scenes where a random sample of a
        self._iteration_mode = 'sample'

        self._binary_bokeh = binary_bokeh
        if self._binary_bokeh:
            print("Using binary bokeh strength of F22.0 and F2.0!")
            self._iteration_mode = 'scene'

        # initialize dir
        self._mode_dir = self._data_path.joinpath(self._mode.value)
        if not self._mode_dir.exists():
            raise FileNotFoundError(f"Mode directory {self._mode_dir} does not exist!")

        # load metadata list
        self._scene_list = sorted([load(open(f)) for f in self._mode_dir.joinpath("metadata").glob("*.json")],
                                  key=lambda x: x['id'])
        if len(self._scene_list) == 0:
            raise FileNotFoundError(f"No metadata files found in {self._mode_dir.joinpath('metadata')}!")

        self._sample_list = []
        for scene_id, metadata in enumerate(self._scene_list):
            for sample_id in range(len(metadata['target_images'])):
                self._sample_list.append((scene_id, sample_id))

        self._device = device

        print(f"RealBokeh Dataloader initialized in {mode} mode with {len(self._scene_list)} scenes and {len(self._sample_list)} samples!")

    def __len__(self):
        # Since for Validation and Test we are iterating over every sample of every scene, we need to return the
        # length of the sample list, otherwise we return the length of the scene list.
        if self._iteration_mode == 'sample':
            return len(self._sample_list)
        else:
            return len(self._scene_list)

    def __getitem__(self, index: int):
        if self._iteration_mode == 'sample':
            metadata: dict = self._scene_list[self._sample_list[index][0]]

            target = Image.open(self._mode_dir.joinpath(metadata['target_images'][self._sample_list[index][1]]))
            tgt_av = metadata['target_avs'][self._sample_list[index][1]]

            target_name = Path(metadata['target_images'][self._sample_list[index][1]]).stem
        else:
            metadata: dict = self._scene_list[index]

            target_idx = 0
            target = Image.open(self._mode_dir.joinpath(metadata['target_images'][target_idx])) if \
                self._mode != Mode.PRED else None
            tgt_av = metadata['target_avs'][target_idx]

            target_name = Path(metadata['target_images'][target_idx]).stem

        source = Image.open(self._mode_dir.joinpath(metadata['source_image']))

        if self.defocus_deblur_mode:
            tmp = source
            source = target
            target = tmp

        # Embed metadata and generate auxiliary maps

        aperture_embedding = calculate_aperture_embedding(tgt_av)

        maps = generate_maps(source, aperture_embedding, target=target)

        return_dict = build_input_dict(maps, aperture_embedding, target_name, self._device)

        return return_dict

class EBB(Dataset):

    def __init__(self, data_path: Union[str, Path], mode: Mode, device: str = 'cuda'):
        self._data_path: Path = Path(data_path) if isinstance(data_path, str) else data_path
        self._mode = mode
        self._device = device

        # initialize dir
        self._mode_dir = self._data_path.joinpath(self._mode.value)
        if not self._mode_dir.exists():
            raise FileNotFoundError(f"Mode directory {self._mode_dir} does not exist!")

        # Initialize input and gt lists
        self._image_list = sorted([Path(f) for f in self._mode_dir.joinpath("in").glob("*.jpg")], key=lambda x: x.stem)
        self._gt_list = sorted([Path(f) for f in self._mode_dir.joinpath("gt").glob("*.jpg")], key=lambda x: x.stem)

        if len(self._image_list) == 0:
            raise FileNotFoundError(f"No images found in {self._mode_dir.joinpath('in')}!")
        if len(self._gt_list) == 0:
            raise FileNotFoundError(f"No images found in {self._mode_dir.joinpath('gt')}!")
        if len(self._image_list) != len(self._gt_list):
            raise ValueError(f"Number of images and gt images do not match in {self._mode_dir}, {len(self._image_list)} inputs != {len(self._gt_list)} gts!")

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, index: int):
        source = crop_to_divisible(Image.open(self._image_list[index]), divisor=4)
        target = crop_to_divisible(Image.open(self._gt_list[index]), divisor=4)

        aperture_embedding = calculate_aperture_embedding(2.0)

        maps = generate_maps(source, aperture_embedding, target=target)

        return_dict = build_input_dict(maps, aperture_embedding, self._image_list[index].name, self._device)

        return return_dict