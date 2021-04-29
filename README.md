## Class Discriminative CutMix (Ongoing)

This repository is a fork of original CutMix implementation from 'clovaai' (https://github.com/clovaai/CutMix-PyTorch)

- Added CUB-200-2011 and MosquitoDL training code
    - Add image transforms and data loaders. 
    - Add learning rate adjustment.

- Added Attentive CutMix (Walawalkar et al, ICASSP 2020 Tech Report)
    - Also implemented not mixed one. (Similar as Cutout)

- Added Class Activation Mapping(CAM) related functions. (Proposed!)
    - Acquire activation maps from the final layer. (layer4, Related to Attentive CutMix)
        - Added callback function for a forward hook.
    - Acquire class-wise activation maps from the final layer.



### Usage

- train_cutout.py

    - --length(int) : Length of one side of the occlusion square. 
```
python3 train_cutout.py --device "3" --dataset mosquitodl --net_type resnet --epochs 90 --batch_size 32 --lr 5e-3 --wd 1e-4 --depth 50 --length 168 --expname R50_Cutout_5e-3_Mosquito_Len168
```

- train_cutmix.py

    - --cutmix_prob (float, [0, 1]): Probability for applying CutMix on given mini-batch.
    - --beta (float): Hyperparameter for $`\alpha`$ determining beta distribution of $`\beta(\alpha, \alpha)`$

```
python3 train_cutmix.py --device "3" --dataset cub200 --net_type resnet --epochs 90 --batch_size 32 --lr 1e-3 --wd 1e-4 --cutmix_prob 0.5 --depth 50 --expname R50_CutMix_1e-3_CUB_P05;

```

- train_attentive_cutmix.py
    - --cut_prob (float, [0, 1]) : Probability for applying Attentive CutMix on given mini-batch.
    - --k (int) : Number of top-k highly activated pixels on the target feature map (Here, 4th block of ResNet50).
    - --image_priority(str, 'A' or 'B')
        - If 'A', the highly activated regions of 'image A' are preserved, and the rest are replaced to 'image B'.
            - Reported as 'Attentive CutMix (Walawalka et al., 2020)
        - If 'B', the highly activated regions of 'image B' are replaced to 'image A', and the rest are preserved.
            - Enforces model confusion. 

```
python3 train_attentive_cutmix.py --device "3" --cut_prob 0.5 --dataset cub200 --net_type resnet --epochs 90 --image_priority B --batch_size 32 --lr 1E-3 --wd 1e-4 --depth 50 --k 15 --expname  R50_AttentiveCutMix_P05_k15_1E-3_ModeB_cub200
```
- train_multiscale_attentive_cutmix.py ----------> __Proposed!__ (Ongoing)
    - Randomly select feature maps in stages.
    - --cut_prob (float, [0, 1]) : Probability for applying Attentive CutMix on given mini-batch.
    - --k (int) : Number of top-k highly activated pixels on the target feature map (Here, 4th block of ResNet50).

- train_multiscale_class_attentive_cutmix.py   ----------->  __Proposed!__ (Ongoing)
    - --cut_prob (float, [0, 1]) : Probability for applying Attentive CutMix on given mini-batch.
    - --k (int) : Number of top-k highly activated pixels on the target feature map (Here, 4th block of ResNet50).
    - --image_priority : Strategy for target image masking
        - Mode A: Preserve highly activated 'A', and replace the regions to image 'B'
        - Mode B: Replace highly activated 'A' to 'B', and preserve the rest region.

- kfold_trainer_variants.py
    - Set of k-fold cross validation sources on refactoring.
- kfold_main.py
    - Set of k-fold cross validation sources on refactoring.