# python3 train_attentive_cutmix.py --device "0" --cut_prob 0.0 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 1 --expname R50_AttentiveCutMix_P00_k1_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P00_k1_1E-3_ModeB_cub200.txt;

python3 train_attentive_cutmix.py --device "0" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 1 --expname R50_AttentiveCutMix_P05_k1_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P05_k1_1E-3_ModeB_cub200.txt \
|
python3 train_attentive_cutmix.py --device "1" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 3 --expname R50_AttentiveCutMix_P05_k3_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P05_k3_1E-3_ModeB_cub200.txt \
|
python3 train_attentive_cutmix.py --device "2" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 6 --expname R50_AttentiveCutMix_P05_k6_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P05_k6_1E-3_ModeB_cub200.txt \
|
python3 train_attentive_cutmix.py --device "3" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 9 --expname R50_AttentiveCutMix_P05_k9_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P05_k9_1E-3_ModeB_cub200.txt;

python3 train_attentive_cutmix.py --device "0" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --image_priority A --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 12 --expname  R50_AttentiveCutMix_P05_k12_1E-3_ModeA_cub200 > R50_AttentiveCutMix_P05_k12_1E-3_ModeA_cub200.txt \
|
python3 train_attentive_cutmix.py --device "1" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --image_priority A --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 15 --expname  R50_AttentiveCutMix_P05_k15_1E-3_ModeA_cub200 > R50_AttentiveCutMix_P05_k15_1E-3_ModeA_cub200.txt \
|
python3 train_attentive_cutmix.py --device "2" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 12 --expname  R50_AttentiveCutMix_P05_k12_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P05_k12_1E-3_ModeB_cub200.txt \
|
python3 train_attentive_cutmix.py --device "3" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 15 --expname  R50_AttentiveCutMix_P05_k15_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P05_k15_1E-3_ModeB_cub200.txt;


# python3 train_attentive_cutmix.py --device "0" --cut_prob 0.1 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 1 --expname R50_AttentiveCutMix_P01_k1_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P01_k1_1E-3_ModeB_cub200.txt \
# |
# python3 train_attentive_cutmix.py --device "1" --cut_prob 0.1 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 3 --expname R50_AttentiveCutMix_P01_k3_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P01_k3_1E-3_ModeB_cub200.txt \
# |
# python3 train_attentive_cutmix.py --device "2" --cut_prob 0.1 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 6 --expname R50_AttentiveCutMix_P01_k6_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P01_k6_1E-3_ModeB_cub200.txt \
# |
# python3 train_attentive_cutmix.py --device "3" --cut_prob 0.1 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 9 --expname R50_AttentiveCutMix_P01_k9_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P01_k9_1E-3_ModeB_cub200.txt;


# python3 train_attentive_cutmix.py --device "2" --cut_prob 0.1 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 12 --expname  R50_AttentiveCutMix_P01_k12_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P01_k12_1E-3_ModeB_cub200.txt \
# |
# python3 train_attentive_cutmix.py --device "3" --cut_prob 0.1 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 15 --expname  R50_AttentiveCutMix_P01_k15_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P01_k15_1E-3_ModeB_cub200.txt;


# python3 train_attentive_cutmix.py --device "0" --cut_prob 0.3 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 1 --expname R50_AttentiveCutMix_P03_k1_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P03_k1_1E-3_ModeB_cub200.txt \
# |
# python3 train_attentive_cutmix.py --device "1" --cut_prob 0.3 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 3 --expname R50_AttentiveCutMix_P03_k3_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P03_k3_1E-3_ModeB_cub200.txt \
# |
# python3 train_attentive_cutmix.py --device "2" --cut_prob 0.3 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 6 --expname R50_AttentiveCutMix_P03_k6_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P03_k6_1E-3_ModeB_cub200.txt \
# |
# python3 train_attentive_cutmix.py --device "3" --cut_prob 0.3 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 9 --expname R50_AttentiveCutMix_P03_k9_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P03_k9_1E-3_ModeB_cub200.txt;


# python3 train_attentive_cutmix.py --device "2" --cut_prob 0.3 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 12 --expname  R50_AttentiveCutMix_P03_k12_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P03_k12_1E-3_ModeB_cub200.txt \
# |
# python3 train_attentive_cutmix.py --device "3" --cut_prob 0.3 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 15 --expname  R50_AttentiveCutMix_P03_k15_1E-3_ModeB_cub200 > R50_AttentiveCutMix_P03_k15_1E-3_ModeB_cub200.txt;


