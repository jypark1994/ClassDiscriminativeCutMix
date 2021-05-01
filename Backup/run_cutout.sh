# LR:1e-3 and WD:1E-4 for CUB200 with 448 pixels, but we use 224 in same settings.

python3 train_cutout.py --device "0" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --length 22 --expname R50_Cutout_1e-3_CUB_Len22 > R50_Cutout_1e-3_CUB_Len22.txt \
|
python3 train_cutout.py --device "1" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --length 56 --expname R50_Cutout_1e-3_CUB_Len56 > R50_Cutout_1e-3_CUB_Len56.txt \
|
python3 train_cutout.py --device "2" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --length 112 --expname R50_Cutout_1e-3_CUB_Len112 > R50_Cutout_1e-3_CUB_Len112.txt \
|
python3 train_cutout.py --device "3" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --depth 50 --length 168 --expname R50_Cutout_1e-3_CUB_Len168 > R50_Cutout_1e-3_CUB_Len168.txt;


python3 train_cutout.py --device "0" --dataset mosquitodl --net_type resnet --epochs 90 --batch_size 32 --lr 5e-3 --wd 1e-4 --depth 50 --length 22 --expname R50_Cutout_5e-3_Mosquito_Len22 > R50_Cutout_5e-3_Mosquito_Len22.txt \
|
python3 train_cutout.py --device "1" --dataset mosquitodl --net_type resnet --epochs 90 --batch_size 32 --lr 5e-3 --wd 1e-4 --depth 50 --length 56 --expname R50_Cutout_5e-3_Mosquito_Len56 > R50_Cutout_5e-3_Mosquito_Len56.txt \
|
python3 train_cutout.py --device "2" --dataset mosquitodl --net_type resnet --epochs 90 --batch_size 32 --lr 5e-3 --wd 1e-4 --depth 50 --length 112 --expname R50_Cutout_5e-3_Mosquito_Len112 > R50_Cutout_5e-3_Mosquito_Len112.txt \
|
python3 train_cutout.py --device "3" --dataset mosquitodl --net_type resnet --epochs 90 --batch_size 32 --lr 5e-3 --wd 1e-4 --depth 50 --length 168 --expname R50_Cutout_5e-3_Mosquito_Len168 > R50_Cutout_5e-3_Mosquito_Len168.txt;