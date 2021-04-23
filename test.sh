# python3 train_cutmix.py --device "0,1,2,3" --dataset cub200 --net_type resnet --epochs 90 --batch_size 128 --lr 1e-3 --wd 1e-4 --cutmix_prob 0.0 --beta 0 --depth 50;

python3 train_attentive_cutmix2.py --device "0,1,2,3" --cut_prob 0.0 --dataset cub200 --net_type resnet --epochs 90 --batch_size 128 --lr 1E-3 --wd 1e-4 --depth 50 --k 1;