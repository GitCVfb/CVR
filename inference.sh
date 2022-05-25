#According to the RS exposure mechanisms in Carla-RS and Fastec-RS datasets, we set
#--load_1st_GS=0 ==> Corresponding to the middle scanline of second RS frame, i.e., t=1.0
#--load_1st_GS=1 ==> Corresponding to the first  scanline of second RS frame, i.e., t=0.5

# !! Updata the path to the dataset and directory to save your trained models with your own local path !!
carla_dataset_type=Carla
carla_root_path_test_data=/home1/fanbin/fan/raw_data/carla/data_test/test/

fastec_dataset_type=Fastec
fastec_root_path_test_data=/home1/fanbin/fan/raw_data/faster/data_test/test/

#Path of our pretrained CVR model   (--model_type=CVR)
model_dir_carla=../model_weights/pretrain_cvr/carla/
model_dir_fastec=../model_weights/pretrain_cvr/fastec/
## or
#Path of our pretrained CVR* model   (--model_type=CVR*)
#model_dir_carla=../model_weights/pretrain_cvr_star/carla/
#model_dir_fastec=../model_weights/pretrain_cvr_star/fastec/

results_dir=/home1/fanbin/fan/CVR/deep_unroll_results/

cd deep_unroll_net


python inference.py \
          --dataset_type=$fastec_dataset_type \
          --dataset_root_dir=$fastec_root_path_test_data \
          --log_dir=$model_dir_fastec \
          --results_dir=$results_dir \
          --crop_sz_H=480 \
          --compute_metrics \
          --model_type=CVR \
          --model_label=pretrain \
          --load_1st_GS=1

#python inference.py \
#          --dataset_type=$carla_dataset_type \
#          --dataset_root_dir=$carla_root_path_test_data \
#          --log_dir=$model_dir_carla \
#          --results_dir=$results_dir \
#          --crop_sz_H=448 \
#          --compute_metrics \
#          --model_type=CVR \
#          --model_label=pretrain \
#          --load_1st_GS=1


