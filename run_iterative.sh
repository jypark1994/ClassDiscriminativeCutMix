
echo "Training MCACM"
python3 kfold_main.py --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode label --cut_mode A --k 5 --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_CamLabel_CutA_K5_P09_L5e-3_W4E-5_9' \
|
python3 kfold_main.py --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode label --cut_mode A --k 5 --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_CamLabel_CutA_K5_P09_L5e-3_W4E-5_10' \
|
python3 kfold_main.py --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode label --cut_mode A --k 5 --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_CamLabel_CutA_K5_P09_L5e-3_W4E-5_11' \
|
python3 kfold_main.py --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode label --cut_mode A --k 5 --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_CamLabel_CutA_K5_P09_L5e-3_W4E-5_12';

echo "Training CACM"
python3 kfold_main.py --single_scale --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode label --cut_mode A --k 20 --cut_prob 0.7 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLabel_CutA_K20_P07_L5e-3_W4E-5_9' \
|
python3 kfold_main.py --single_scale --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode label --cut_mode A --k 20 --cut_prob 0.7 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLabel_CutA_K20_P07_L5e-3_W4E-5_10' \
|
python3 kfold_main.py --single_scale --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode label --cut_mode A --k 20 --cut_prob 0.7 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLabel_CutA_K20_P07_L5e-3_W4E-5_11' \
|
python3 kfold_main.py --single_scale --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode label --cut_mode A --k 20 --cut_prob 0.7 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLabel_CutA_K20_P07_L5e-3_W4E-5_12';

echo "Training MACM"
python3 kfold_main.py --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MACM" --cam_mode label --cut_mode A --k 10 --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_MACM_CamLabel_CutA_K10_P09_L5e-3_W4E-5_9' \
|
python3 kfold_main.py --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MACM" --cam_mode label --cut_mode A --k 10 --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_MACM_CamLabel_CutA_K10_P09_L5e-3_W4E-5_10' \
|
python3 kfold_main.py --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MACM" --cam_mode label --cut_mode A --k 10 --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_MACM_CamLabel_CutA_K10_P09_L5e-3_W4E-5_11' \
|
python3 kfold_main.py --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MACM" --cam_mode label --cut_mode A --k 10 --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_MACM_CamLabel_CutA_K10_P09_L5e-3_W4E-5_12';

echo "Training ACM"
python3 kfold_main.py --single_scale --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MACM" --cam_mode label --cut_mode A --k 10 --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_ACM_CamLabel_CutA_K10_P01_L5e-3_W4E-5_9' \
|
python3 kfold_main.py --single_scale --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MACM" --cam_mode label --cut_mode A --k 10 --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_ACM_CamLabel_CutA_K10_P01_L5e-3_W4E-5_10' \
|
python3 kfold_main.py --single_scale --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MACM" --cam_mode label --cut_mode A --k 10 --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_ACM_CamLabel_CutA_K10_P01_L5e-3_W4E-5_11' \
|
python3 kfold_main.py --single_scale --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MACM" --cam_mode label --cut_mode A --k 10 --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_ACM_CamLabel_CutA_K10_P01_L5e-3_W4E-5_12';

echo "Training CutMix"
python3 kfold_main.py --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "cutmix" --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_CutMix_P01_L5e-3_W4E-5_9' \
|
python3 kfold_main.py --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "cutmix" --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_CutMix_P01_L5e-3_W4E-5_10' \
|
python3 kfold_main.py --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "cutmix" --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_CutMix_P01_L5e-3_W4E-5_11' \
|
python3 kfold_main.py --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "cutmix" --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_CutMix_P01_L5e-3_W4E-5_12';


echo "Training Vanilla"
python3 kfold_main.py --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "vanilla" \
    --flag_vervose --expr_name 'R50_Mosquito_vanilla_P01_L5e-3_W4E-5_9' \
|
python3 kfold_main.py --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "vanilla" \
    --flag_vervose --expr_name 'R50_Mosquito_vanilla_P01_L5e-3_W4E-5_10' \
|
python3 kfold_main.py --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "vanilla" \
    --flag_vervose --expr_name 'R50_Mosquito_vanilla_P01_L5e-3_W4E-5_11' \
|
python3 kfold_main.py --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "vanilla" \
    --flag_vervose --expr_name 'R50_Mosquito_vanilla_P01_L5e-3_W4E-5_12';