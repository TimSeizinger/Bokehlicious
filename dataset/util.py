from enum import Enum
from math import ceil, floor

from PIL import Image
from numpy import array
from torch import Tensor, linspace, meshgrid, tensor, cat, float32, full
from torchvision.transforms.functional import to_tensor
from cv2 import resize, INTER_CUBIC
from pathlib import Path

class Mode(Enum):
    TRAIN = 'train'
    VAL = 'validation'
    TEST = 'test'
    PRED = 'prediction'

def load_image(img_path, target_av=2.0, max_dim=2000, interpolation=INTER_CUBIC, min_divisor=4, device='cuda') -> dict:
    """
    Loads an image and prepares it for input to the model
    :param img_path: path to the image
    :param target_av: target aperture value between 20.0 and 2.0
    :param max_dim: maximum dimension of the image, less than 2000 is recommended
    :param sampling: downsampling method
    :param min_divisor: ensures that the network can process the image without tensor size mismatches
    :return: Dictionary that can be used as an input to the Bokehlicious model
    """
    img = Image.open(img_path)

    img = crop_to_divisible(img, min_divisor)

    img = downsample(img, max_dim, interpolation)

    aperture_embedding = calculate_aperture_embedding(target_av)

    maps = generate_maps(img, aperture_embedding)

    input_dict = build_input_dict(maps, aperture_embedding, Path(img_path).stem, device=device)

    input_dict["source"] = input_dict["source"].unsqueeze(0)
    input_dict["target"] = input_dict["target"].unsqueeze(0)
    input_dict["bokeh_strength"] = input_dict["bokeh_strength"].unsqueeze(0)
    input_dict["pos_map"] = input_dict["pos_map"].unsqueeze(0)
    input_dict["bokeh_strength_map"] = input_dict["bokeh_strength_map"].unsqueeze(0)

    return input_dict

def calculate_aperture_embedding(tgt_av: float, max_av: float = 2.0):
    return max_av / tgt_av

def generate_maps(source: Image, aperture_embedding, target: Image = None):

    # Generate auxiliary maps

    pos_map = get_pos_map(*source.size)

    bokeh_strength_map = get_map(*source.size, bokeh_strength=aperture_embedding)

    return {'source': source, 'target': target,
            'pos_x': pos_map[0].unsqueeze(0),
            'pos_y': pos_map[1].unsqueeze(0),
            'bokeh_strength_map': bokeh_strength_map,
            }

def get_pos_map(w: int, h: int) -> (Tensor, Tensor):
    if w > h:
        crop_dist = (1 - (h / w)) / 2
        x_lin = linspace(0, 1, w)
        y_lin = linspace(1 - crop_dist, crop_dist, h)
        return meshgrid(x_lin, y_lin, indexing='xy')
    elif w < h:
        crop_dist = (1 - (w / h)) / 2
        x_lin = linspace(crop_dist, 1 - crop_dist, w)
        y_lin = linspace(1, 0, h)
        return meshgrid(x_lin, y_lin, indexing='xy')
    else:  # w == h
        x_lin = linspace(0, 1, w)
        y_lin = linspace(1, 0, h)
        return meshgrid(x_lin, y_lin, indexing='xy')

def build_input_dict(maps: dict, aperture_embedding: float, target_name: str, device='cuda'):
    return {
        "source": to_tensor(maps['source']).to(device),
        "target": to_tensor(maps['target']).to(device) if maps['target'] is not None else tensor(0.).to(device),
        "bokeh_strength": tensor(aperture_embedding, dtype=float32, device=device),
        "pos_map": cat((maps['pos_x'], maps['pos_y']), dim=0).to(device),
        "bokeh_strength_map": maps['bokeh_strength_map'].to(device),
        "image_name": target_name
    }

def get_map(w: int, h: int, bokeh_strength: float) -> Tensor:
    return full((1, h, w), bokeh_strength, dtype=float32)

def crop_to_divisible(img: Image, divisor: int):
    width, height = img.size
    return center_crop(img, width - (width % divisor), height - (height % divisor))

def center_crop(img:Image, new_width, new_height):
    width, height = img.size

    if new_width is None:
        new_width = min(width, height)
    if new_height is None:
        new_height = min(width, height)

    left = int(ceil((width - new_width) / 2))
    right = width - int(floor((width - new_width) / 2))
    top = int(ceil((height - new_height) / 2))
    bottom = height - int(floor((height - new_height) / 2))

    return img.crop([left, top, right, bottom])

def downsample(img: Image, max_dim: int, interpolation) -> Image:
    if img.size[0] > max_dim or img.size[1] > max_dim:
        cvImg = array(img)
        rescaled_res = get_resolution(height=cvImg.shape[0], width=cvImg.shape[1], longest_side=max_dim)
        cvImg = resize(cvImg, rescaled_res, interpolation=interpolation)
        return Image.fromarray(cvImg)
    else:
        return img

def get_resolution(height, width, longest_side):
    if height > width:
        rescaled_height = longest_side
        rescaled_width = int(width * (longest_side / height))
    else:
        rescaled_width = longest_side
        rescaled_height = int(height * (longest_side / width))
    return rescaled_width, rescaled_height
