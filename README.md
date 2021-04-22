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