import torch
from torch import nn
import numpy as np

def print_v(str_t, vervose=False):
    if vervose:
        print(str_t)

def rand_bbox(size, lam): 
    '''
        From ClovaAi
    '''
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def generate_attentive_mask(attention_map, top_k):
    """
        Author: Junyoung Park (jy_park@inu.ac.kr)
        Input:
            attention_map   (Tensor) : NxWxH tensor after GAP.
            top_k           (Tensor) : Number of candidates of the most intense points.
        Output:
            mask            (Tensor) : NxWxH tensor for masking attentive regions
            coords          (Tensor) : Normalized coordinates(cx, cy) for masked regions
    """
    N, W, H = attention_map.shape

    x = attention_map.reshape([N, W *H])

    _, indices = torch.sort(x, descending=True, dim=1)
    top_indices = indices[:, :top_k] # [N, Top_k]

    mask = torch.ones_like(x)

    for i in range(N):
        mask[i, top_indices[i]] = 0
    
    mask = mask.reshape([N, W, H])
    # print_v(mask)

    return mask

class Wrapper(nn.Module):
    '''
        Author: Junyoung Park (jy_park@inu.ac.kr)
    '''
    def __init__(self, model, stage_names):
        super(Wrapper, self).__init__()

        self.dict_activation = {}
        self.dict_gradients = {}
        self.forward_hook_handles = []
        self.backward_hook_handles = []

        self.net = model
        self.stage_names = stage_names
        self.num_stages = len(self.stage_names)

        def forward_hook_function(name): # Hook function for the forward pass.
            def get_class_activation(module, input, output):
                self.dict_activation[name] = output.data
            return get_class_activation

        def backward_hook_function(name): # Hook function for the backward pass.
            def get_class_gradient(module, input, output):
                self.dict_gradients[name] = output
            return get_class_gradient

        for L in self.stage_names:
            for k, v in self.net.named_modules():
                if L in k:
                    self.forward_hook_handles.append(v.register_forward_hook(forward_hook_function(L)))
                    self.backward_hook_handles.append(v.register_backward_hook(backward_hook_function(L)))
                    print(f"Registered forward/backward hook on \'{k}\'")
                    break

    def forward(self, x):
        self.clear_dict()
        return self.net(x)
            
    def print_current_dicts(self):
        for k, v in self.dict_activation.items():
            print("[FW] Layer:", k)
            print("[FW] Shape:", v.shape)
        for k, v in self.dict_gradients.items():
            print("[BW] Layer:", k)      
            print("[BW] Shape:", v.shape)

    def clear_dict(self):
        for k, v in self.dict_activation.items():
            v = None
        for k, v in self.dict_gradients.items():
            v = None