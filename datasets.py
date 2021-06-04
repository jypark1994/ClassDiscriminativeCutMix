import os

import torch
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Resize
from transforms_imagenet import *
import utils
from cub200 import CUB200

def IP102(root, crop_size=224, batch_size=(64,64), num_workers=8):
    # Transforms: https://www.kaggle.com/mekouaryoussef/rendufinal
    # Official Settings (in CVPR2019)
    # BS = 64
    # LR = 0.01 * 0.1 @ 40 EPOCHS
    # WD = 5E-4, M = 0.9
    # Input_size = 224

    if isinstance(batch_size, tuple):
        bs_train = batch_size[0]
        bs_test = batch_size[1]
    else:
        bs_train, bs_test = (batch_size, batch_size)

    transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
    ])
    transforms_test = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
    ])

    train_dataset = datasets.ImageFolder(os.path.join(root,'train'), transform=transforms_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs_train, shuffle=True, num_workers=num_workers)

    valid_dataset = datasets.ImageFolder(os.path.join(root,'val'), transform=transforms_test)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs_test, shuffle=False, num_workers=num_workers)

    test_dataset = datasets.ImageFolder(os.path.join(root,'test'), transform=transforms_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs_test, shuffle=False, num_workers=num_workers)

    num_classes = 102


    return train_loader, valid_loader, test_loader, num_classes


def MosquitoDL_fold(root, crop_size=224, num_folds=5, batch_size=(64, 32), num_workers=8, ver='v2'):
    '''
        Author: Junyoung Park (jy_park@inu.ac.kr)
        
        Mosquito Classification DataLoader

        num_folds(int): Use training data split with 'num_folds' for k-fold cross validation.
        crop_size(Tuple or int): if tuple, (bs_train, bs_test). if int, use bs_train = bs_test.

    '''

    if isinstance(batch_size, tuple):
        bs_train = batch_size[0]
        bs_test = batch_size[1]
    else:
        bs_train, bs_test = (batch_size, batch_size)

    # %%
    init_scale = 1.15

    if ver == 'v1':
        transforms_train = transforms.Compose([
            transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0.2,hue=0.1),
            transforms.RandomAffine(360,scale=[init_scale-0.15, init_scale+0.15]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.816, 0.744, 0.721],std=[0.146, 0.134, 0.121]),
        ])
    elif ver == 'v2':
        transforms_train = transforms.Compose([
            transforms.RandomAffine(360,scale=[0.5, 0.8]),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0.2,hue=0.1),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
    elif ver == 'v3':
        transforms_train = transforms.Compose([
            transforms.RandomAffine(360,scale=[0.8, 1.2]),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.2,contrast=0.4,saturation=0.4,hue=0.2),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
        ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.816, 0.744, 0.721],std=[0.146, 0.134, 0.121]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(root,'train'), transform=transforms_train)

    len_fold, len_fold_rest = len(train_dataset)//num_folds, len(train_dataset) % num_folds

    fold_lengths = [len_fold for x in range(num_folds)]

    if(len_fold_rest != 0):
        fold_lengths.pop()
        fold_lengths.append(len_fold_rest + len_fold)

    print(f"Dataset with length {len(train_dataset)} divided into {fold_lengths} (Sum:{sum(fold_lengths)})")

    train_dataset = torch.utils.data.random_split(train_dataset, fold_lengths)

    train_loader = {x: torch.utils.data.DataLoader(train_dataset[x], bs_train,
                                                shuffle=True, num_workers=num_workers)
                    for x in range(num_folds)}

    test_dataset = datasets.ImageFolder(os.path.join(root,'valid'), transform=transforms_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=num_workers)

    num_classes = 6

    return train_loader, test_loader, num_classes

def CIFAR_loaders(root, type='10',batch_size=(64, 32), num_workers=4):

    if isinstance(batch_size, tuple):
        bs_train = batch_size[0]
        bs_test = batch_size[1]
    else:
        bs_train, bs_test = (batch_size, batch_size)

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if type == '100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root, train=True, download=True, transform=transform_train),
            batch_size=bs_train, shuffle=True, num_workers=num_workers, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root, train=False, transform=transform_test),
            batch_size=bs_test, shuffle=True, num_workers=num_workers, pin_memory=True)
        num_classes = 100
    elif type == '10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root, train=True, download=True, transform=transform_train),
            batch_size=bs_train, shuffle=True, num_workers=num_workers, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root, train=False, transform=transform_test),
            batch_size=bs_test, shuffle=True, num_workers=num_workers, pin_memory=True)
        num_classes = 10
    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))
        
    return train_loader, valid_loader, num_classes

def ImageNet_loaders(root, batch_size=(64, 32), num_workers=4):
    '''
        Modified from the original implementation by ClovaAI
    '''

    if isinstance(batch_size, tuple):
        bs_train = batch_size[0]
        bs_test = batch_size[1]
    else:
        bs_train, bs_test = (batch_size, batch_size)

    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    jittering = ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.4)
    lighting = Lighting(alphastd=0.1,
                                eigval=[0.2175, 0.0188, 0.0045],
                                eigvec=[[-0.5675, 0.7192, 0.4009],
                                        [-0.5808, -0.0045, -0.8140],
                                        [-0.5836, -0.6948, 0.4203]])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            jittering,
            lighting,
            normalize,
        ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs_train, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs_test, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    num_classes = 1000

    return train_loader, val_loader, num_classes

def CUB200_loaders(root, crop_size=224, batch_size=(64,32), num_workers=4):
    
    if isinstance(batch_size, tuple):
        bs_train = batch_size[0]
        bs_test = batch_size[1]
    else:
        bs_train, bs_test = (batch_size, batch_size)

    num_classes = 200

    # Apply same transforms as 'https://github.com/zhangyongshun/resnet_finetune_cub'

    train_transforms = transforms.Compose([
            transforms.Resize(int(crop_size*1.1429)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
    ])
    val_transforms = transforms.Compose([
            transforms.Resize(int(crop_size*1.1429)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
    ])

    train_dataset = CUB200(root, transform = train_transforms, train=True, download=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs_train, shuffle=True, num_workers=num_workers)
    val_dataset = CUB200(root, transform = val_transforms, train=False, download=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs_test, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader, num_classes

# def CUB200_loaders(root, crop_size=224, batch_size=(64,32), num_workers=4):

#     if isinstance(batch_size, tuple):
#         bs_train = batch_size[0]
#         bs_test = batch_size[1]
#     else:
#         bs_train, bs_test = (batch_size, batch_size)

#     num_classes = 200

#     # Apply same transforms as 'https://github.com/zhangyongshun/resnet_finetune_cub'

#     train_transforms = transforms.Compose([
#             transforms.RandomResizedCrop(crop_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                  std=(0.229, 0.224, 0.225))
#     ])
#     val_transforms = transforms.Compose([
#             transforms.Resize(crop_size),
#             transforms.CenterCrop(crop_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                  std=(0.229, 0.224, 0.225))
#     ])

#     train_dataset = CUB200(root, transform = train_transforms, train=True, download=False)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs_train, shuffle=True, num_workers=num_workers)
#     val_dataset = CUB200(root, transform = val_transforms, train=False, download=False)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs_test, shuffle=True, num_workers=num_workers)

#     return train_loader, val_loader, num_classes

def MosquitoDL_loaders(root, crop_size=224, batch_size=(64, 32), num_workers=4):
    '''
        Author: Junyoung Park (jy_park@inu.ac.kr)
        
        Mosquito Classification DataLoader

        crop_size(Tuple or int): if tuple, (bs_train, bs_test). if int, use bs_train = bs_test.

    '''

    if isinstance(batch_size, tuple):
        bs_train = batch_size[0]
        bs_test = batch_size[1]
    else:
        bs_train, bs_test = (batch_size, batch_size)

    # %%
    init_scale = 1.15
    transforms_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0.2,hue=0.1),
        transforms.RandomAffine(360,scale=[init_scale-0.15, init_scale+0.15]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.816, 0.744, 0.721],std=[0.146, 0.134, 0.121]),
    ])

    transforms_test = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.816, 0.744, 0.721],std=[0.146, 0.134, 0.121]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(root,'train'), transform=transforms_train)

    print(f"Dataset with length {len(train_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, bs_train,
                                                shuffle=True, num_workers=num_workers)

    test_dataset = datasets.ImageFolder(os.path.join(root,'valid'), transform=transforms_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs_test, shuffle=False, num_workers=num_workers)

    num_classes = 6

    return train_loader, test_loader, num_classes