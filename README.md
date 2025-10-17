# Bokehlicious: <br/> Photorealistic Bokeh Rendering with Controllable Apertures

[Tim Seizinger](https://scholar.google.com/citations?user=PKdW78wAAAAJ), [Florin-Alexandru Vasluianu](https://scholar.google.com/citations?user=kHHzuyoAAAAJ), [Marcos V. Conde](https://mv-lab.github.io/), [Zongwei Wu](https://scholar.google.com/citations?user=3QSALjX498QC), [Radu Timofte](https://www.cvlai.net/)

[Computer Vision Lab](https://www.informatik.uni-wuerzburg.de/computervision/) , [CAIDAS](https://www.caidas.uni-wuerzburg.de/), [University of Wurzburg](https://www.uni-wuerzburg.de/en/), Germany

Official Code and Dataset repository of Bokehlicious: Photorealistic Bokeh Rendering with Controllable Apertures

## Installation

```
git clone https://github.com/TimSeizinger/Bokehlicious.git
cd Bokehlicious
pip install -r requirements.txt
```

## Usage

predict.py lets you run Bokehlicious on a single image.
For example:
```
python predict.py -img_path ./examples/collie.jpg -size small -av 2.8
```

Here _-img\_path_ is the path to the image you want to render, _-size_ is the size of the model you want to use (small or large) and _-av_ is the aperture f-stop to control the strength of bokeh (between 2.0 and 20.0).

## Evaluation

This Repository includes evaluation scripts for Bokeh Rendering on our new RealBokeh dataset, as well as EBB! Val294 and EBB400.

Before running the evaluation script you need to download the [test set](https://huggingface.co/datasets/timseizinger/RealBokeh_3MP/tree/main/test) of RealBokeh and copy it to the ./dataset/RealBokeh folder.
The same applies to EBB! Val294 and EBB400.

To run the evaluation script use:
```
python evaluate.py -dataset RealBokeh -size small --save_outputs
```

Here _-dataset_ is the dataset you want to evaluate on (RealBokeh, RealBokeh_bin, EBB_Val294, EBB400), _-size_ is the size of the model you want to use (small or large) and _--save_outputs_ is a flag to save the rendered images.

## RealBokeh Dataset

You can find our RealBokeh Dataset on Huggingface!

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-lg.svg)](https://huggingface.co/datasets)

## Citation

If you find our work useful for your research work please cite:

```
@inproceedings{seizinger2025bokehlicious,
  author    = {Seizinger, Tim and Vasluianu, Florin-Alexandru and Conde, Marcos and Wu, Zongwei and Timofte, Radu},
  title     = {Bokehlicious: Photorealistic Bokeh Rendering with Controllable Apertures},
  booktitle = {ICCV},
  year      = {2025},
}
```