echo "Set 1"

python3 main.py --gpus '0' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 1 --cut_prob 0.1 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K1_P01" \
|
python3 main.py --gpus '1' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 3 --cut_prob 0.1 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K3_P01" \
|
python3 main.py --gpus '2' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 6 --cut_prob 0.1 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K6_P01" \
|
python3 main.py --gpus '3' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 9 --cut_prob 0.1 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K9_P01";

echo "Set 2"

python3 main.py --gpus '0' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 1 --cut_prob 0.3 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K1_P03" \
|
python3 main.py --gpus '1' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 3 --cut_prob 0.3 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K3_P03" \
|
python3 main.py --gpus '2' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 6 --cut_prob 0.3 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K6_P03" \
|
python3 main.py --gpus '3' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 9 --cut_prob 0.3 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K9_P03";

echo "Set 1-2"

python3 main.py --gpus '0' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 12 --cut_prob 0.1 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K12_P01" \
|
python3 main.py --gpus '1' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 12 --cut_prob 0.3 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K12_P03" \
|
python3 main.py --gpus '2' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 15 --cut_prob 0.1 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K15_P01" \
|
python3 main.py --gpus '3' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 15 --cut_prob 0.3 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K15_P03";

echo "Set 3"

python3 main.py --gpus '0' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 1 --cut_prob 0.5 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K1_P05" \
|
python3 main.py --gpus '1' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 3 --cut_prob 0.5 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K3_P05" \
|
python3 main.py --gpus '2' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 6 --cut_prob 0.5 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K6_P05" \
|
python3 main.py --gpus '3' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 9 --cut_prob 0.5 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K9_P05";

echo "Set 4"

python3 main.py --gpus '0' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 1 --cut_prob 0.7 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K1_P07" \
|
python3 main.py --gpus '1' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 3 --cut_prob 0.7 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K3_P07" \
|
python3 main.py --gpus '2' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 6 --cut_prob 0.7 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K6_P07" \
|
python3 main.py --gpus '3' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 9 --cut_prob 0.7 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K9_P07";

echo "Set 3-4"

python3 main.py --gpus '0' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 12 --cut_prob 0.5 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K12_P05" \
|
python3 main.py --gpus '1' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 12 --cut_prob 0.7 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K12_P07" \
|
python3 main.py --gpus '2' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 15 --cut_prob 0.5 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K15_P05" \
|
python3 main.py --gpus '3' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 15 --cut_prob 0.7 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K15_P07";

echo "Set 5"

python3 main.py --gpus '0' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 1 --cut_prob 0.9 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K1_P09" \
|
python3 main.py --gpus '1' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 3 --cut_prob 0.9 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K3_P09" \
|
python3 main.py --gpus '2' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 6 --cut_prob 0.9 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K6_P09" \
|
python3 main.py --gpus '3' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 9 --cut_prob 0.9 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K9_P09";

echo "Set 5"

python3 main.py --gpus '1' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 12 --cut_prob 0.9 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K12_P09" \
|
python3 main.py --gpus '2' --data_type "cub200" --single_scale --pretrained --batch_size 32 --crop_size 224 --train_mode "MCACM" --k 15 --cut_prob 0.9 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "Modified_R50_CUB200_S224_CACM_K15_P09";