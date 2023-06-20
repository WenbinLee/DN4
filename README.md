# DN4 in PyTorch (2023 Version)

We provide a PyTorch implementation of DN4 for few-shot learning. If you use this code for your research, please cite: 

[Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning](http://cs.nju.edu.cn/rl/people/liwb/CVPR19.pdf).<br> 
[Wenbin Li](https://cs.nju.edu.cn/liwenbin/), Lei Wang, Jinglin Xu, Jing Huo, Yang Gao and Jiebo Luo. In CVPR 2019.<br> 
<img src='imgs/Flowchart.bmp' width=600/>


## Prerequisites
- Linux
- Python 3.8
- Pytorch 1.7.0
- GPU + CUDA CuDNN
- pillow, torchvision, scipy, numpy

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/WenbinLee/DN4.git
cd DN4
```

- Install [PyTorch](http://pytorch.org) 1.0 and other dependencies.

### Datasets
[Caltech-UCSD Birds-200-2011](https://data.caltech.edu/records/20098), [Standford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [Standford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html), [*mini*ImageNet](https://arxiv.org/abs/1606.04080v2) and [*tiered*ImageNet](https://arxiv.org/abs/1803.00676) are available at [Google Drive](https://drive.google.com/drive/u/1/folders/1SEoARH5rADckI-_gZSQRkLclrunL-yb0) and [百度网盘(提取码：yr1w)](https://pan.baidu.com/s/1M3jFo2OI5GTOpytxgtO1qA).


###  miniImageNet Few-shot Classification
- Train a 5-way 1-shot model based on Conv64F or ResNet256F:
```bash
python DN4_Train_5way1shot.py --dataset_dir ./datasets/miniImageNet --data_name miniImageNet
or
python DN4_Train_5way1shot_Resnet.py --dataset_dir ./datasets/miniImageNet --data_name miniImageNet
```
- Test the model (specify the dataset_dir, basemodel, and data_name first):
```bash
python DN4_Test_5way1shot.py --resume ./results/DN4_miniImageNet_Conv64F_5Way_1Shot_K3/model_best.pth.tar --basemodel Conv64F
or
python DN4_Test_5way1shot.py --resume ./results/DN4_miniImageNet_ResNet256F_5Way_1Shot_K3/model_best.pth.tar --basemodel ResNet256F
```

- The results on the miniImageNet dataset (If you set neighbor_k as 1, you may get better results in some cases): 
<img src='imgs/Results_miniImageNet2.bmp' align="center" width=900>



## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{DN4_CVPR_2019,
  author       = {Wenbin Li and
                  Lei Wang and
                  Jinglin Xu and
                  Jing Huo and
                  Yang Gao and
                  Jiebo Luo},
  title        = {Revisiting Local Descriptor Based Image-To-Class Measure for Few-Shot Learning},
  booktitle    = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages        = {7260--7268},
  year         = {2019}
}
```
