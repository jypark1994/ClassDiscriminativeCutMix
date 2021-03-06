# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import pyramidnet as PYRM
import utils
import numpy as np

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import warnings

from cub200 import CUB200

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Attentive CutMix PyTorch CUB-200, MosqutoDL Training')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1E-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5E-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=1, type=float,
                    help='hyperparameter beta')
parser.add_argument('--device', default='0', type=str,
                    help='Target GPU for computation')
parser.add_argument('--pretrained', default='./pretrained/R50_ImageNet_Baseline.pth', type=str,
                    help='Pretrained *.pth path')

parser.add_argument('--k', default=3, type=int, help='Number of most activated patches on the final layer.')
parser.add_argument('--cut_prob', default=0, type=float, help='Attentive CutMix probability')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100

dict_activation = {}

def generate_attentive_mask(attention_map, top_k):
    """
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

    top_indices = indices[:, :top_k]

    cell_width, cell_height = 1/W, 1/H

    rows, cols = (top_indices//W)/N, (top_indices%W)/N
    cx = cell_width/2 + rows*cell_width
    cy = cell_height/2 + cols*cell_height
    coords = torch.cat((cx, cy), dim=0).T

    # print(cx, cy)
    # print(coords)

    mask = x.clone()

    for i in range(N):
        mask[i, top_indices[i]] = 0

    mask = mask.reshape([N, W, H])
    # print(mask)
    mask[mask != 0] = 1

    return mask, coords

def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    if args.dataset.startswith('cifar'):
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

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))
    elif args.dataset == 'mosquitodl':
        transforms_train = transforms.Compose([
            transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0.2,hue=0.1),
            transforms.RandomAffine(360,scale=[1.55, 1.9]),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.816, 0.744, 0.721],std=[0.146, 0.134, 0.121]),
        ])

        transforms_test = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        train_dataset = datasets.ImageFolder("../mosquitoClassification/MosquitoDL/train", transform=transforms_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

        val_dataset = datasets.ImageFolder("../mosquitoClassification/MosquitoDL/valid", transform=transforms_test)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

        numberofclass = 6
    elif args.dataset == 'cub200':
        numberofclass = 200
        train_transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        val_transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        train_dataset = CUB200("../mosquitoClassification", transform = train_transforms, train=True, download=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_dataset = CUB200("../mosquitoClassification", transform = val_transforms, train=False, download=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)


    elif args.dataset == 'imagenet':
        traindir = os.path.join('/home/data/ILSVRC/train')
        valdir = os.path.join('/home/data/ILSVRC/val')
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
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        numberofclass = 1000

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained != 'scratch':
        print(f"Load pretrained weights from \'{args.pretrained}\'.")

        pretrained_dict = torch.load(args.pretrained)['state_dict']
        new_model_dict = model.state_dict()

        for k, v in new_model_dict.items():
            if 'fc' in k:
                continue
            else:
                new_model_dict[k] = v

        model.load_state_dict(new_model_dict)

    

    def get_class_activation(name, input, output):
        dict_activation['layer4'] = output.data

    for k, v in model.named_modules():
        if 'layer4' in k:
            v.register_forward_hook(get_class_activation)
            print(f"Registered forward hook on \'{k}\'")
            break

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True

    for epoch in range(0, args.epochs):

        adjust_learning_rate(optimizer, epoch)
        
        epoch_t_start = time.time()

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        epoch_t_end = time.time() - epoch_t_start

        print(f'- Epoch time: {epoch_t_end:.4f}[sec]')
        print(f'- Estimated time left: {epoch_t_end*(args.epochs - epoch):.4f}[sec]')
        print('-'*30)

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)



def train(train_loader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        _, _, W, H = input.shape

        target = target.cuda()

        r = 1

        if False:
            # print("Generate Mask")
            # print(f"Apply CutMix at r={r:.2f} < {args.cut_prob:.2f}")
            # compute feature maps
            output = model(input)

            # TODO: Acquire activation maps from the final layer.
            final_fmap = dict_activation['layer4'].mean(dim=1) # Shape: [N x W_f x H_f]

            # dict_activation['layer4'] = None # Initialize activation hook
            N, W_f, H_f = final_fmap.shape

            # Visualizing Activation Map
            # ========== Uncomment to put visualization =========== 
            # final_fmap_to_visualize = final_fmap.unsqueeze(1).repeat(1,3,1,1)
            # cam_grid = make_grid(final_fmap_to_visualize.detach().cpu(), nrow=4).permute([1,2,0])
            # plt.imshow(cam_grid)
            # plt.axis('off')
            # plt.savefig('[Debug] Visualize Activation.png')
            # exit()
            # =====================================================

            # Acquire Highly Activated Region Informations (Grid based)
            #   - Consideration: Allow or not to select multiple patches ... (grid-based?, contour detection?)
            #   - Attentive CutMix(2020) uses highly activated top N patches from 7x7 grid map.
            #   - Generate Normalized BBox coordnates (x_min, y_min, x_max, y_max)

            attention_masks, _ = generate_attentive_mask(final_fmap, top_k = args.k) # Grid-based, Masking Top k patches
            # In tech report, they tested top_k 1~15, and suggested 6 is proper.
            # attention_masks: [N, W_f, H_f]
            # coords: [[cx_1, cy_1] ... [cx_k, cy_k]]
            # print(attention_masks.shape)

            upsampled_attention_masks = F.interpolate(attention_masks.unsqueeze(1).repeat([1,3,1,1]), 
                size=input.shape[-2:], mode='nearest') # [N, W_f, H_f] -> [N, 1, W_f, H_f] -> [N, 3, W_f, H_f] -> [N, C, W, H]
            # print(upsampled_attention_masks.shape)

            occluded_batch = input * upsampled_attention_masks #[N, C, W, H] * [N, C, W, H]
            # print(occluded_batch.shape)

            n_occluded_pixels = args.k
            n_total_pixels = W*H
            
            lam = 1 - (n_occluded_pixels/n_total_pixels) # 1 - Mixed Ratio

            rand_index = torch.randperm(input.size()[0]).cuda() # Randomly select image sample to pasted from the batch
            target_a = target
            target_b = target[rand_index]
            
            input = (1 - upsampled_attention_masks * input) * input[rand_index]

            optimizer.zero_grad()

            output = model(input)

            loss = criterion(output, target_a)  * (1 - lam) + criterion(output, target_b) * lam

        else:
            # print("Bypass")

            output = model(input)

            loss = criterion(output, target)

        if i % 100 == 0 and epoch % 10 == 0:
            cam_grid = make_grid(input.detach().cpu(), normalize=True, nrow=8).permute([1,2,0])
            plt.imshow(cam_grid)
            plt.title(f"Attentive CutMix Batch Sample at k={args.k}")
            plt.axis('off')
            plt.savefig(os.path.join('./runs/',args.expname, f'Occluded Batch K{args.k}_E{epoch}_B{i}.png'))

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def rand_bbox(size, lam):
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


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

    elif args.dataset == ('cub200'):
        lr = args.lr * (0.1 ** (epoch // 30))

    elif args.dataset == ('mosquitodl'):
        lr = args.lr * (0.1 ** (epoch // 50))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print(correct)
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
