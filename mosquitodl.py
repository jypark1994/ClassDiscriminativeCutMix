import os

from torchvision import datasets, transforms

class Bypass(object):
    """ Bypass input image.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        """
            Input:
                img (Tensor): Input image.
            Return:
                img (Tensor): Input image.
        """

        return img

def MosquitoDL(root, batch_size=32, num_workers=8, cutout=False,normalize=False):
    init_scale = 1.15

    transforms_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0.2,hue=0.1),
        transforms.RandomAffine(360,scale=[init_scale-0.15,init_scale+0.4]),
        transforms.Resize(224),
        transforms.RandomCrop(224), 
        # In 2020, we used center cropping for MosquitoDL, but we replaced to random cropping to prevent further overfitting.
        transforms.ToTensor(),
        Cutout(n_holes=args.n_holes, length=args.length) if cutout else Bypass(),
        transforms.Normalize(mean=[0.816, 0.744, 0.721],std=[0.146, 0.134, 0.121]) if normalize else Bypass(),
    ])

    transforms_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(root,'train'), transform=transforms_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = datasets.ImageFolder(os.path.join(root,'valid'), transform=transforms_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    numberofclass = 6