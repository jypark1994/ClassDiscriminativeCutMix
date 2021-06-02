python3 main.py --gpus '0' --data_type "IP102" --dataset_root "~/datasets/IP102_Splitted/" --pretrained \
    --batch_size 64 --crop_size 224 --num_epochs 120 --train_mode "Vanilla" \
    --learning_rate 1e-2 --weight_decay 5e-4 --scheduler_step 40 --expr_name "R50_IP102_S224_Vanilla";