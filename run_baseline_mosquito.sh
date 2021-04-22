python3 train.py --device "0" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 1e-3 --cutmix_prob 0.0 --depth 50 --expname R50_CutMix_1e-3_Mosquito_P00 > R50_CutMix_L1e-3_W1E-3_Mosquito_P00.txt \
|
python3 train.py --device "0" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 5e-4 --cutmix_prob 0.0 --depth 50 --expname R50_CutMix_1e-3_Mosquito_P00 > R50_CutMix_L1e-3_1e-3_Mosquito_P00.txt \
|
python3 train.py --device "0" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 1e-4 --cutmix_prob 0.0 --depth 50 --expname R50_CutMix_1e-3_Mosquito_P00 > R50_CutMix_L1e-3_1e-3_Mosquito_P00.txt \
|
python3 train.py --device "0" --dataset mosquitodl --net_type resnet --epochs 100 --batch_size 32 --lr 1e-3 --wd 4e-5 --cutmix_prob 0.0 --depth 50 --expname R50_CutMix_1e-3_Mosquito_P00 > R50_CutMix_L1e-3_1e-3_Mosquito_P00.txt;