
echo "Train CutMix"
python3 main.py --gpus '0' --data_type "cub200" --pretrained --batch_size 10 --crop_size 448 --train_mode "Vanilla" --learning_rate 1e-2 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_Vanilla" \
|
python3 main.py --gpus '1' --data_type "cub200" --pretrained --batch_size 10 --crop_size 448 --train_mode "CutMix"  --cut_prob 0.3 --learning_rate 1e-2 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_CutMix_P03" \
|
python3 main.py --gpus '0' --data_type "cub200" --pretrained --batch_size 10 --crop_size 448 --train_mode "CutMix"  --cut_prob 0.5 --learning_rate 1e-2 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_CutMix_P05" \
|
python3 main.py --gpus '1' --data_type "cub200" --pretrained --batch_size 10 --crop_size 448 --train_mode "CutMix"  --cut_prob 0.7 --learning_rate 1e-2 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_CutMix_P07";

echo "Train CutMix and Vanilla"

python3 main.py --gpus '0' --data_type "cub200" --pretrained --batch_size 10 --crop_size 448 --train_mode "CutMix" --cut_prob 0.9 --learning_rate 1e-2 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_CutMix_P09" \
|
python3 main.py --gpus '1' --data_type "cub200" --pretrained --batch_size 10 --crop_size 448 --train_mode "CutMix"  --cut_prob 0.1 --learning_rate 1e-2 --weight_decay 1e-4 --expr_name "R50_CUB200_S448_CutMix_P01";
