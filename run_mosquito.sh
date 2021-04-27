python3 train_cutout.py --pretrained '' --device "0" --dataset mosquitodl --net_type resnet --epochs 150 --batch_size 64 --lr 1e-2 --wd 1e-5 --depth 18 --length 56 --expname R18_Mosquito_1e-2_Len56 > R18_Mosquito_1e-2_Len56.txt \
|
python3 train_cutmix.py --pretrained '' --device "1" --dataset mosquitodl --net_type resnet --epochs 150 --batch_size 64 --lr 1e-2 --wd 1e-5 --cutmix_prob 0 --depth 18 --expname R18_Baseline_1e-2_Mosquito > R18_Baseline_1e-2_Mosquito.txt \
|
python3 train_cutmix.py --pretrained '' --device "2" --dataset mosquitodl --net_type resnet --epochs 150 --batch_size 64 --lr 1e-2 --wd 1e-5 --cutmix_prob 0.5 --depth 18 --expname R18_CutMix_1e-2_Mosquito_P05 > R18_CutMix_1e-2_Mosquito_P05.txt \
|
python3 train_multiscale_class_attentive_cutmix.py --pretrained '' --target_mode 'label' --image_priority A --device "3" --cut_prob 0.5 --dataset mosquitodl --net_type resnet --epochs 150 --batch_size 64 --lr 1e-2 --wd 1e-5 --depth 18 --k 6 --expname R18_MS_Class_Attentivecutmix_k1_1e-2_P05_Mosquito > R18_MS_Class_Attentivecutmix_k6_1e-2_P05_Mosquito.txt \
