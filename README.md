# Context-Aware Video Reconstruction for Rolling Shutter Cameras

This repository contains the source code for the paper: [Context-Aware Video Reconstruction for Rolling Shutter Cameras (CVPR2022)](https://arxiv.org/pdf/2205.12912.pdf).
Given two rolling shutter frames at adjacent times 0 and 1, the proposed CVR can synthesize a high-quality intermediate global shutter frame corresponding to any time 0<t<1, i.e., generating a smooth and coherent global shutter video.

From left to right: Overlayed rolling shutter images, recovered global shutter videos by RSSR ([ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_Inverting_a_Rolling_Shutter_Camera_Bring_Rolling_Shutter_Images_to_ICCV_2021_paper.pdf)) and our CVR (this paper), respectively.

![fountain_overlay](/result_demo/fountain_overlay.png) ![fountain_rssr](/result_demo/fountain_rssr.gif) ![fountain_cvr](/result_demo/fountain_cvr.gif)
![bus_overlay](/result_demo/bus_overlay.png) ![bus_rssr](/result_demo/bus_rssr.gif) ![bus_cvr](/result_demo/bus_cvr.gif)

## Installation
Install the dependent packages:
```
pip install -r requirements.txt
```
The code is tested with PyTorch 1.6.0 with CUDA 10.2.89.

Note that the baseline of our CVR network comes from [RSSR](https://github.com/GitCVfb/RSSR). 
Similarly, we first need to configure the following packages:

#### Install correlation package
```
cd ./package_correlation
python setup.py install
```
#### Install differentiable forward warping package
```
cd ./package_forward_warp
python setup.py install
```
#### Install core package
```
cd ./package_core
python setup.py install
```
#### Install reblur_package
```
cd ./reblur_package
python setup.py install
```
## Demo with our pretrained model
Please download the [pretrained model](https://drive.google.com/drive/folders/11aciusk4wBfKffgoflywKVZTpDW_QdtS?usp=sharing), including network models of [RSSR](https://github.com/GitCVfb/RSSR), CVR, and CVR*, respectively. Then unzip these three subfolders to the `model_weights` folder of the main directory.

You can now test our method with the provided images in the `demo` folder.

Note that our CVR can be tested directly.
To test CVR*, you need to change the weight's path (--log_dir) and the model's type (--model_type) in files `demo.sh`, `demo_video.sh`, and `inferencce.sh`.

To generate the global shutter images corresponding to time steps 0.5 and 1.0, simply run
```
sh demo.sh
```
To generate multiple global shutter video frames (stored in .fig format), e.g. 10× temporal upsampling, please run
```
sh demo_video.sh
```
The visualization results will be stored in the `experiments` folder. Note that additional examples in the dataset can be tested similarly.

## Datasets
- **Carla-RS** and **Fastec-RS:** Download these two datasets to your local computer from [here](https://github.com/ethliup/DeepUnrollNet).

## Training and evaluating
You can run following commands to re-train the network.
```
# !! Please update the corresponding paths in 'train_carla.sh' and 'train_fastec.sh' with  #
# !! your own local paths, before run following command!!      #

sh train_carla.sh
sh train_fastec.sh
```

You can run following commands to obtain the quantitative evaluations.
```
# !! Please update the path to test data in 'inference.sh'
# !! with your own local path, before run following command!!

sh inference.sh
```
Note that `--load_1st_GS=0` denotes the correction evaluation corresponding to time 1.0, and `--load_1st_GS=1` denotes the correction evaluation corresponding to time 0.5.

## Citations
Please cite our paper if necessary:
```
@inproceedings{fan_CVR_CVPR22,
  title={Context-Aware Video Reconstruction for Rolling Shutter Cameras},
  author={Fan, Bin and Dai, Yuchao and Zhang, Zhiyuan and Liu, Qi and He, Mingyi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

@inproceedings{fan_RSSR_ICCV21,
  title={Inverting a rolling shutter camera: bring rolling shutter images to high framerate global shutter video},
  author={Fan, Bin and Dai, Yuchao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4228--4237},
  year={2021}
}
```

## Statement
This project is for research purpose only, please contact us for the licence of commercial use. For any other questions or discussion please contact: binfan@mail.nwpu.edu.cn
