echo "Set 1"

python3 main.py --gpus '0' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 5 --cut_prob 0.1 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K5_P01" \
|
python3 main.py --gpus '1' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 10 --cut_prob 0.1 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K10_P01" \
|
python3 main.py --gpus '2' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 15 --cut_prob 0.1 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K15_P01" \
|
python3 main.py --gpus '3' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 20 --cut_prob 0.1 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K20_P01";

echo "Set 2"

python3 main.py --gpus '0' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 5 --cut_prob 0.3 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K5_P03" \
|
python3 main.py --gpus '1' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 10 --cut_prob 0.3 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K10_P03" \
|
python3 main.py --gpus '2' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 15 --cut_prob 0.3 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K15_P03" \
|
python3 main.py --gpus '3' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 20 --cut_prob 0.3 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K20_P03";

echo "Set 3"

python3 main.py --gpus '0' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 5 --cut_prob 0.5 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K5_P05" \
|
python3 main.py --gpus '1' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 10 --cut_prob 0.5 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K10_P05" \
|
python3 main.py --gpus '2' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 15 --cut_prob 0.5 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K15_P05" \
|
python3 main.py --gpus '3' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 20 --cut_prob 0.5 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K20_P05";

echo "Set 4"

python3 main.py --gpus '0' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 5 --cut_prob 0.7 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K5_P07" \
|
python3 main.py --gpus '1' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 10 --cut_prob 0.7 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K10_P07" \
|
python3 main.py --gpus '2' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 15 --cut_prob 0.7 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K15_P07" \
|
python3 main.py --gpus '3' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 20 --cut_prob 0.7 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K20_P07";

echo "Set 5"

python3 main.py --gpus '0' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 5 --cut_prob 0.9 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K5_P09" \
|
python3 main.py --gpus '1' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 10 --cut_prob 0.9 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K10_P09" \
|
python3 main.py --gpus '2' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 15 --cut_prob 0.9 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K15_P09" \
|
python3 main.py --gpus '3' --data_type "cub200" --pretrained --batch_size 16 --crop_size 448 --train_mode "MCACM" --k 20 --cut_prob 0.9 --learning_rate 1e-3 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_MCACM_K20_P09";