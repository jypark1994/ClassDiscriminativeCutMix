# LR:1e-3 and WD:1E-4 for CUB200 with 448 pixels, but we use 224 in same settings.

echo "CutOut Training set 1"

python3 train_cutout.py --device "0" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --length 22 --expname R50_Cutout_1e-3_CUB_Len22 > R50_Cutout_1e-3_CUB_Len22.txt \
|
python3 train_cutout.py --device "1" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --length 56 --expname R50_Cutout_1e-3_CUB_Len56 > R50_Cutout_1e-3_CUB_Len56.txt \
|
python3 train_cutout.py --device "2" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --length 112 --expname R50_Cutout_1e-3_CUB_Len112 > R50_Cutout_1e-3_CUB_Len112.txt \
|
python3 train_cutout.py --device "3" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --length 168 --expname R50_Cutout_1e-3_CUB_Len168 > R50_Cutout_1e-3_CUB_Len168.txt;

# echo "CutMix Training set 3"

# python3 train.py --device "0" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 5e-4 --cutmix_prob 0.0 --depth 50 --expname R50_CutMix_1e-3_Mosquito_P00 > R50_CutMix_1e-3_Mosquito_P00.txt \
# |
# python3 train.py --device "1" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 5e-4 --cutmix_prob 0.1 --depth 50 --expname R50_CutMix_1e-3_Mosquito_P01 > R50_CutMix_1e-3_Mosquito_P01.txt \
# |
# python3 train.py --device "2" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 5e-4 --cutmix_prob 0.3 --depth 50 --expname R50_CutMix_1e-3_Mosquito_P03 > R50_CutMix_1e-3_Mosquito_P03.txt \
# |
# python3 train.py --device "3" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 5e-4 --cutmix_prob 0.5 --depth 50 --expname R50_CutMix_1e-3_Mosquito_P05 > R50_CutMix_1e-3_Mosquito_P05.txt;

# echo "CutMix Training set 4"

# python3 train.py --device "0" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 5e-4 --cutmix_prob 0.7 --depth 50 --expname R50_CutMix_1e-3_Mosquito_P07 > R50_CutMix_1e-3_Mosquito_P07.txt \
# |
# python3 train.py --device "1" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 5e-4 --cutmix_prob 0.9 --depth 50 --expname R50_CutMix_1e-3_Mosquito_P07 > R50_CutMix_1e-3_Mosquito_P09.txt;


# echo "CutMix Training set 1"

# python3 train.py --device "0" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --cutmix_prob 0.0 --depth 50 --expname R50_CutMix_1e-3_CUB_P00 > R50_CutMix_1e-3_CUB_P00.txt \
# |
# python3 train.py --device "1" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --cutmix_prob 0.1 --depth 50 --expname R50_CutMix_1e-3_CUB_P01 > R50_CutMix_1e-3_CUB_P01.txt \
# |
# python3 train.py --device "2" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --cutmix_prob 0.3 --depth 50 --expname R50_CutMix_1e-3_CUB_P03 > R50_CutMix_1e-3_CUB_P03.txt \
# |
# python3 train.py --device "3" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --cutmix_prob 0.5 --depth 50 --expname R50_CutMix_1e-3_CUB_P05 > R50_CutMix_1e-3_CUB_P05.txt;

# echo "CutMix Training set 2"

# python3 train.py --device "0" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --cutmix_prob 0.7 --depth 50 --expname R50_CutMix_1e-3_CUB_P07 > R50_CutMix_1e-3_CUB_P07.txt \
# |
# python3 train.py --device "1" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --cutmix_prob 0.9 --depth 50 --expname R50_CutMix_1e-3_CUB_P09 > R50_CutMix_1e-3_CUB_P09.txt;


echo "CutOut Training set 2"

python3 train_cutout.py --device "0" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 5e-4 --depth 50 --length 22 --expname R50_Cutout_1e-3_Mosquito_Len22 > R50_Cutout_1e-3_Mosquito_Len22.txt \
|
python3 train_cutout.py --device "1" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 5e-4 --depth 50 --length 56 --expname R50_Cutout_1e-3_Mosquito_Len56 > R50_Cutout_1e-3_Mosquito_Len56.txt \
|
python3 train_cutout.py --device "2" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 5e-4 --depth 50 --length 112 --expname R50_Cutout_1e-3_Mosquito_Len112 > R50_Cutout_1e-3_Mosquito_Len112.txt \
|
python3 train_cutout.py --device "3" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 5e-4 --depth 50 --length 168 --expname R50_Cutout_1e-3_Mosquito_Len168 > R50_Cutout_1e-3_Mosquito_Len168.txt;




