# hSMAL: Horse specific version of SMAL model

Code for loading the hSMAL model and pose prior of the hSMAL model.

P.S: The hSMAL model is 100% compatible with the [SMAL](https://smal.is.tue.mpg.de/) model. Please check the [paper](https://arxiv.org/abs/2106.10102) for more details. 

[Project Page](https://sites.google.com/view/cv4horses/cv4horses)

[paper](https://arxiv.org/abs/2106.10102)


### Requirements
- Python 3.6
- [PyTorch](https://pytorch.org/) tested on version `1.7.1`

### Installation

#### Setup Conda Environment
```
conda create -n hsmal python==3.8
conda activate 
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Install hSMAL model
download the [hSMAL model](https://sites.google.com/view/cv4horses) and place the files in the directory ```smpl_models```.

### Use

Test with the model. Run ```python smal_torch.py```.

Test the pose prior of the model. Run ```python pose_prior.py```.



### Notes
This repository is widely based on [smalst](https://github.com/silviazuffi/smalst).


### Citation

If you use this code please cite
```
@article{li2021hsmal,
  title={hSMAL: Detailed horse shape and pose reconstruction for motion pattern recognition},
  author={Li, Ci and Ghorbani, Nima and Broom{\'e}, Sofia and Rashid, Maheen and Black, Michael J and Hernlund, Elin and Kjellstr{\"o}m, Hedvig and Zuffi, Silvia},
  journal={arXiv preprint arXiv:2106.10102},
  year={2021}
}
```



