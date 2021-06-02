echo "Set 1"

python3 kfold_main.py --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.1 --cut_mode B --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P01_T01_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.3 --cut_mode B  --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P01_T03_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.5 --cut_mode B  --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P01_T05_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.7 --cut_mode B  --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P01_T07_L5e-3_W4E-5';

echo "Set 2"

python3 kfold_main.py --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.1 --cut_mode B --cut_prob 0.3 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P03_T01_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.3 --cut_mode B  --cut_prob 0.3 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P03_T03_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.5 --cut_mode B  --cut_prob 0.3 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P03_T05_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.7 --cut_mode B  --cut_prob 0.3 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P03_T07_L5e-3_W4E-5';

echo "Set 3"

python3 kfold_main.py --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.1 --cut_mode B --cut_prob 0.5 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P05_T01_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.3 --cut_mode B  --cut_prob 0.5 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P05_T03_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.5 --cut_mode B  --cut_prob 0.5 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P05_T05_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.7 --cut_mode B  --cut_prob 0.5 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P05_T07_L5e-3_W4E-5';

echo "Set 4"

python3 kfold_main.py --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.1 --cut_mode B --cut_prob 0.7 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P07_T01_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.3 --cut_mode B  --cut_prob 0.7 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P07_T03_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.5 --cut_mode B  --cut_prob 0.7 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P07_T05_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.7 --cut_mode B  --cut_prob 0.7 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P07_T07_L5e-3_W4E-5';

echo "Set 5"

python3 kfold_main.py --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.1 --cut_mode B --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P09_T01_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.3 --cut_mode B  --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P09_T03_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.5 --cut_mode B  --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P09_T05_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.7 --cut_mode B  --cut_prob 0.9 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P09_T07_L5e-3_W4E-5';


echo "Set 6"

python3 kfold_main.py --gpus 0 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.9 --cut_mode B --cut_prob 0.1 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P01_T09_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 1 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.9 --cut_mode B  --cut_prob 0.3 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P03_T09_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 2 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.9 --cut_mode B  --cut_prob 0.5 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P05_T09_L5e-3_W4E-5' \
|
python3 kfold_main.py --gpus 3 --net_type resnet50 --pretrained --num_epochs 25 --num_folds 5 \
    --batch_size 32 --learning_rate 5e-3 --weight_decay 4e-5 --scheduler_step 25 \
    --train_mode "MCACM" --cam_mode likely --mask_mode thres --threshold 0.9 --cut_mode B  --cut_prob 0.7 \
    --flag_vervose --expr_name 'R50_Mosquito_MCACM_Camlikely_CutB_P07_T09_L5e-3_W4E-5';