import os

import torch
from torchvision import datasets, transforms
from transforms_imagenet import *

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

    jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.4)
    lighting = utils.Lighting(alphastd=0.1,
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
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
    ])
    val_transforms = transforms.Compose([
            transforms.Resize(int(crop_size/0.875)),
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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs_test, shuffle=True, num_workers=num_workers)

    num_classes = 6

    return train_loader, test_loader, num_classes