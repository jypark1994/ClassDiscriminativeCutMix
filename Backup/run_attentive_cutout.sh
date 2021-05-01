# echo "Attentive CutOut P=0.1 (1)"

# python3 train_attentive_cutout.py --device "0" --cut_prob 0.1 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 1 --expname R50_AttentiveCutOut_k1_1e-3_P01_cub200 > R50_AttentiveCutOut_k1_1e-3_P01_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "1" --cut_prob 0.1 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 3 --expname R50_AttentiveCutOut_k3_1e-3_P01_cub200 > R50_AttentiveCutOut_k3_1e-3_P01_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "2" --cut_prob 0.1 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 6 --expname R50_AttentiveCutOut_k6_1e-3_P01_cub200 > R50_AttentiveCutOut_k6_1e-3_P01_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "3" --cut_prob 0.1 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 9 --expname R50_AttentiveCutOut_k9_1e-3_P01_cub200 > R50_AttentiveCutOut_k9_1e-3_P01_cub200.txt;

# echo "Attentive CutOut P=0.1 (2) and"

# echo "Attentive CutOut P=0.3 (2)"

# python3 train_attentive_cutout.py --device "0" --cut_prob 0.1 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 12 --expname  R50_AttentiveCutOut_k12_1e-3_P01_cub200 > R50_AttentiveCutOut_k12_1e-3_P01_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "1" --cut_prob 0.1 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 15 --expname  R50_AttentiveCutOut_k15_1e-3_P01_cub200 > R50_AttentiveCutOut_k15_1e-3_P01_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "2" --cut_prob 0.3 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 12 --expname  R50_AttentiveCutOut_k12_1e-3_P03_cub200 > R50_AttentiveCutOut_k12_1e-3_P03_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "3" --cut_prob 0.3 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 15 --expname  R50_AttentiveCutOut_k15_1e-3_P03_cub200 > R50_AttentiveCutOut_k15_1e-3_P03_cub200.txt;

# echo "Attentive CutOut P=0.3 (1)"

# python3 train_attentive_cutout.py --device "0" --cut_prob 0.3 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 1 --expname R50_AttentiveCutOut_k1_1e-3_P03_cub200 > R50_AttentiveCutOut_k1_1e-3_P03_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "1" --cut_prob 0.3 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 3 --expname R50_AttentiveCutOut_k3_1e-3_P03_cub200 > R50_AttentiveCutOut_k3_1e-3_P03_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "2" --cut_prob 0.3 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 6 --expname R50_AttentiveCutOut_k6_1e-3_P03_cub200 > R50_AttentiveCutOut_k6_1e-3_P03_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "3" --cut_prob 0.3 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 9 --expname R50_AttentiveCutOut_k9_1e-3_P03_cub200 > R50_AttentiveCutOut_k9_1e-3_P03_cub200.txt;


# echo "Attentive CutOut P=0.5 (1)"

# python3 train_attentive_cutout.py --device "0" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 1 --expname R50_AttentiveCutOut_k1_1e-3_P05_cub200 > R50_AttentiveCutOut_k1_1e-3_P05_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "1" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 3 --expname R50_AttentiveCutOut_k3_1e-3_P05_cub200 > R50_AttentiveCutOut_k3_1e-3_P05_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "2" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 6 --expname R50_AttentiveCutOut_k6_1e-3_P05_cub200 > R50_AttentiveCutOut_k6_1e-3_P05_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "3" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 9 --expname R50_AttentiveCutOut_k9_1e-3_P05_cub200 > R50_AttentiveCutOut_k9_1e-3_P05_cub200.txt;

# echo "Attentive CutOut P=0.5 (2)"

# python3 train_attentive_cutout.py --device "2" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 12 --expname  R50_AttentiveCutOut_k12_1e-3_P05_cub200 > R50_AttentiveCutOut_k12_1e-3_P05_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "3" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 15 --expname  R50_AttentiveCutOut_k15_1e-3_P05_cub200 > R50_AttentiveCutOut_k15_1e-3_P05_cub200.txt;

# echo "Attentive CutOut P=0.7 (1)"

# python3 train_attentive_cutout.py --device "0" --cut_prob 0.7 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 1 --expname R50_AttentiveCutOut_k1_1e-3_P07_cub200 > R50_AttentiveCutOut_k1_1e-3_P07_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "1" --cut_prob 0.7 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 3 --expname R50_AttentiveCutOut_k3_1e-3_P07_cub200 > R50_AttentiveCutOut_k3_1e-3_P07_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "2" --cut_prob 0.7 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 6 --expname R50_AttentiveCutOut_k6_1e-3_P07_cub200 > R50_AttentiveCutOut_k6_1e-3_P07_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "3" --cut_prob 0.7 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 9 --expname R50_AttentiveCutOut_k9_1e-3_P07_cub200 > R50_AttentiveCutOut_k9_1e-3_P07_cub200.txt;


# echo "Attentive CutOut P=0.7 (2) and"

# echo "Attentive CutOut P=0.9 (2)"

# python3 train_attentive_cutout.py --device "0" --cut_prob 0.7 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 12 --expname  R50_AttentiveCutOut_k12_1e-3_P07_cub200 > R50_AttentiveCutOut_k12_1e-3_P07_cub200.txt \
# |
# python3 train_attentive_cutout.py --device "1" --cut_prob 0.7 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 15 --expname  R50_AttentiveCutOut_k15_1e-3_P07_cub200 > R50_AttentiveCutOut_k15_1e-3_P07_cub200.txt \
# |
python3 train_attentive_cutout.py --device "2" --cut_prob 0.9 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 12 --expname  R50_AttentiveCutOut_k12_1e-3_P09_cub200 > R50_AttentiveCutOut_k12_1e-3_P09_cub200.txt \
|
python3 train_attentive_cutout.py --device "3" --cut_prob 0.9 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 15 --expname  R50_AttentiveCutOut_k15_1e-3_P09_cub200 > R50_AttentiveCutOut_k15_1e-3_P09_cub200.txt;


echo "Attentive CutOut P=0.9 (1)"

python3 train_attentive_cutout.py --device "0" --cut_prob 0.9 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 1 --expname R50_AttentiveCutOut_k1_1e-3_P09_cub200 > R50_AttentiveCutOut_k1_1e-3_P09_cub200.txt \
|
python3 train_attentive_cutout.py --device "1" --cut_prob 0.9 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 3 --expname R50_AttentiveCutOut_k3_1e-3_P09_cub200 > R50_AttentiveCutOut_k3_1e-3_P09_cub200.txt \
|
python3 train_attentive_cutout.py --device "2" --cut_prob 0.9 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 6 --expname R50_AttentiveCutOut_k6_1e-3_P09_cub200 > R50_AttentiveCutOut_k6_1e-3_P09_cub200.txt \
|
python3 train_attentive_cutout.py --device "3" --cut_prob 0.9 --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --k 9 --expname R50_AttentiveCutOut_k9_1e-3_P09_cub200 > R50_AttentiveCutOut_k9_1e-3_P09_cub200.txt;


