# python3 kfold_main.py --single_scale --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
#     --batch_size 32 --learning_rate 1e-4 --weight_decay 4e-5 --scheduler_step 25 \
#     --train_mode 'vanilla' --flag_vervose --expr_name 'R50_Mosquito_Vanilla_L1E-4_W4E-5' \
# |
# python3 kfold_main.py --single_scale --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
#     --batch_size 32 --learning_rate 5e-4 --weight_decay 4e-5 --scheduler_step 25 \
#     --train_mode 'vanilla' --flag_vervose --expr_name 'R50_Mosquito_Vanilla_L5E-4_W4E-5' \
# |
# python3 kfold_main.py --single_scale --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
#     --batch_size 32 --learning_rate 1e-3 --weight_decay 4e-5 --scheduler_step 25 \
#     --train_mode 'vanilla' --flag_vervose --expr_name 'R50_Mosquito_Vanilla_L1e-3_W4E-5' \
# |
# python3 kfold_main.py --single_scale --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
#     --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
#     --train_mode 'vanilla' --flag_vervose --expr_name 'R50_Mosquito_Vanilla_L5E-3_W4E-5';


echo "Set 1"

python3 kfold_main.py --single_scale --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 5 --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K5_P01_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 10 --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K10_P01_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 15 --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K15_P01_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 20 --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K20_P01_L5e-3_W4E-5';

echo "Set 2"

python3 kfold_main.py --single_scale --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 5 --cut_prob 0.3 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K5_P03_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 10 --cut_prob 0.3 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K10_P03_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 15 --cut_prob 0.3 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K15_P03_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 20 --cut_prob 0.3 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K20_P03_L5e-3_W4E-5';

echo "Set 3"

python3 kfold_main.py --single_scale --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 5 --cut_prob 0.5 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K5_P05_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 10 --cut_prob 0.5 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K10_P05_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 15 --cut_prob 0.5 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K15_P05_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 20 --cut_prob 0.5 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K20_P05_L5e-3_W4E-5';

echo "Set 4"

python3 kfold_main.py --single_scale --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 5 --cut_prob 0.7 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K5_P07_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 10 --cut_prob 0.7 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K10_P07_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 15 --cut_prob 0.7 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K15_P07_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 20 --cut_prob 0.7 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K20_P07_L5e-3_W4E-5';

echo "Set 5"

python3 kfold_main.py --single_scale --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 5 --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K5_P09_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 10 --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K10_P09_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 15 --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K15_P09_L5e-3_W4E-5' \
|
python3 kfold_main.py --single_scale --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --cut_mode A --k 20 --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_CACM_CamLikely_CutA_K20_P09_L5e-3_W4E-5';
