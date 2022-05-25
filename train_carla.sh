#!/bin/bash

# !! Updata the path to the dataset and directory to save your trained models with your own local path !!
carla_dataset_type=Carla
carla_root_path_training_data=/home1/fanbin/fan/raw_data/carla/data_train/train/

log_dir=/home1/fanbin/fan/CVR/deep_unroll_weights/
log_dir_rssr=/home1/fanbin/fan/CVR/model_weights/pretrain_rssr/carla/
#
cd deep_unroll_net


python train_CVR.py \
          --dataset_type=$carla_dataset_type \
          --dataset_root_dir=$carla_root_path_training_data \
          --log_dir=$log_dir \
          --log_dir_rssr=$log_dir_rssr \
          --lamda_L1=10 \
          --lamda_L1_ccl=5 \
          --lamda_perceptual=1 \
          --lamda_flow_smoothness=0.1 \
          --crop_sz_H=448 \
          --model_type=CVR \
          #--continue_train=True \
          #--start_epoch=51 \
          #--model_label=50