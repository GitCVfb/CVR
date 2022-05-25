#!/bin/bash

# create an empty folder for experimental results
mkdir -p experiments/results_demo_fastec
mkdir -p experiments/results_demo_carla

cd deep_unroll_net

#Path of our pretrained CVR model   (--model_type=CVR)
model_dir_carla=../model_weights/pretrain_cvr/carla/
model_dir_fastec=../model_weights/pretrain_cvr/fastec/
## or
#Path of our pretrained CVR* model   (--model_type=CVR*)
#model_dir_carla=../model_weights/pretrain_cvr_star/carla/
#model_dir_fastec=../model_weights/pretrain_cvr_star/fastec/


python inference_demo.py \
            --model_label='pretrain' \
            --results_dir=../experiments/results_demo_carla \
            --data_dir='../demo/Carla' \
            --model_type=CVR \
            --is_Fastec=0 \
            --log_dir=$model_dir_carla


python inference_demo.py \
            --model_label='pretrain' \
            --results_dir=../experiments/results_demo_fastec \
            --data_dir='../demo/Fastec' \
            --model_type=CVR \
            --is_Fastec=1 \
            --log_dir=$model_dir_fastec