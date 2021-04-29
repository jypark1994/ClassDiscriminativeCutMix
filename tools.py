import torch

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

def print_v(str_t, vervose=False):
    if vervose:
        print(str_t)
