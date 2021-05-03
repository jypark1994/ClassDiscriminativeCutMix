import torch
from torch import optim, nn
from torchvision import models

import os
import sys
import csv
import time
import argparse

from trainer import train, train_MACM, train_MCACM, test
from datasets import CIFAR_loaders, MosquitoDL_loaders, ImageNet_loaders, CUB200_loaders 
from utils import Wrapper

parser = argparse.ArgumentParser(description='Train and Evaluate MosquitoDL')
parser.add_argument('--net_type', default='resnet50', type=str,
                    help='networktype: resnet')
parser.add_argument('--data_type', default='cifar10', type=str,
                    help='cifar10, cifar100, cub200, imagenet, mosquitoDL')                    
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--gpus', type=str, default='0')

parser.add_argument('--num_epochs', type=int, default=100)

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--learning_rate', type=float, default=5e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--scheduler_step', type=int, default=30)

parser.add_argument('--expr_name', type=str, default="default")
parser.add_argument('--dataset_root', type=str, default="~/datasets")
parser.add_argument('--flag_vervose', action='store_true')
parser.add_argument('--single_scale', action='store_true')

parser.add_argument('--train_mode', type=str, default="vanilla")
parser.add_argument('--cut_mode', type=str, default="A")
parser.add_argument('--cam_mode', type=str, default="label")
parser.add_argument('--k', type=int, default=1)
parser.add_argument('--cut_prob', type=float, default=0.5)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_root = args.dataset_root
save_path = os.path.join("./results", args.expr_name)

crop_size = args.crop_size # Default
net_type = args.net_type.lower()
data_type = args.data_type.lower()
train_mode = args.train_mode.lower()
num_epochs = args.num_epochs
batch_size = (args.batch_size, args.batch_size)
num_workers = args.num_workers
flag_vervose = args.flag_vervose
single_scale = args.single_scale
scheduler_step = args.scheduler_step
target_mode = args.cam_mode

os.makedirs(save_path, exist_ok=True)
f_print = open(os.path.join(save_path, 'output.txt'), 'w')
sys.stdout = f_print # Change the standard output to the file we created.
print(args)


print(f"Building Dataloaders: {data_type}")

if data_type == 'mosquitodl':
    train_loader, valid_loader, num_classes = MosquitoDL_loaders(dataset_root, crop_size, batch_size, num_workers)
elif 'cifar' in data_type:
    if data_type == 'cifar10':
        train_loader, valid_loader, num_classes = CIFAR_loaders(dataset_root, '10', batch_size, num_workers)
    elif data_type == 'cifar100':
        train_loader, valid_loader, num_classes = CIFAR_loaders(dataset_root, '100', batch_size, num_workers)
    else:
        assert f'Unrecognized \'{data_type}\' for CIFAR dataset.'
elif data_type == 'imagenet':
    train_loader, valid_loader, num_classes = ImageNet_loaders(dataset_root, batch_size, num_workers)
elif data_type == 'cub200':
    train_loader, valid_loader, num_classes = CUB200_loaders(dataset_root, crop_size, batch_size, num_workers)
else:
    assert f'Unsupported Dataset Type \'{data_type}\'.'

print(" - Done !")


print(f"Buliding \'{net_type}\' network...")

if data_type == 'cifar': # Using 32x32 version of resnet. (ClovaAI Implementation)
    pass
else: # Using ImageNet version of resnet.
    if net_type == 'resnet50':
        model = models.resnet50(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = nn.DataParallel(model)
    else:
        assert 'Unsupported Network Type !'

if 'resnet' in net_type:
    stage_names = ['layer1','layer2','layer3','layer4']
elif net_type == 'mobilenetv2':
    stage_names = ['features.2','features.4','features.7','features.14']
else:
    assert "Unsupported network type !"

if single_scale: # using only the final feature maps.
        stage_names = stage_names[-1]

model = Wrapper(model, stage_names) # Wrapper for registering hooks for 'stage_names' of the 'model'.

model = model.to(device)
print(f"\t - Done !")


print("Building Optimizer Related Objects")
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
print(f"\t - Done !")
# %%

logs = []
logs.append(['epoch', 'loss_tr', 'acc_tr', 'loss_val', 'acc_val', 'loss_test', 'acc_test'])
elapsed_time = 0
best_model = None
best_valid_acc = 0

print("==== Training/Evaluation Start ! ====")

for epoch in range(num_epochs):
    print(f"==== Current Epoch: {epoch+1}")

    epoch_start_t = time.time()

    print(f"\t - Train/Val Phase ...")

    if train_mode == 'vanilla':
        model, epoch_train_loss, epoch_train_acc = \
            train(model, train_loader, optimizer, scheduler, criterion, epoch, device, flag_vervose=flag_vervose, \
            net_type='resnet', save_path=save_path)
    elif train_mode == 'cutmix':
        model, epoch_train_loss, epoch_train_acc = \
            train_CutMix(model, train_loader, optimizer, scheduler, criterion, epoch, device, flag_vervose=flag_vervose, \
            net_type='resnet', cut_prob=args.cut_prob, save_path=save_path)
    elif train_mode == 'macm': # Multiscale Attentive Cutmix
        model, epoch_train_loss, epoch_train_acc = \
            train_MACM(model, train_loader, optimizer, scheduler, criterion, epoch, device, flag_vervose=flag_vervose, \
            net_type='resnet', k=args.k, image_priority=args.cut_mode, cut_prob=args.cut_prob, save_path=save_path, target_mode='label')
    elif train_mode == 'mcacm': # Multiscale Class Attentive Cutmix
        model, epoch_train_loss, epoch_train_acc = \
            train_MCACM(model, train_loader, optimizer, scheduler, criterion, epoch, device, flag_vervose=flag_vervose, \
            net_type='resnet', k=args.k, image_priority=args.cut_mode, cut_prob=args.cut_prob, cam_mode=args.cam_mode, save_path=save_path, target_mode='label')
    else:
        assert 'Invalid training mode !'

    print(f"\t - Epoch training loss : {epoch_train_loss:.4f}")
    print(f"\t - Epoch training accuracy : {epoch_train_acc*100:.4f}%")

    print(f"\t - Validation Phase ...")
    model, epoch_valid_loss, epoch_valid_acc = test(model, valid_loader, criterion, device, save_path, epoch)
    print(f"\t - Epoch validation loss : {epoch_valid_loss:.4f}")
    print(f"\t - Epoch validation accuracy : {epoch_valid_acc*100:.4f}%")

    logs.append([epoch, epoch_train_loss, epoch_train_acc, \
        epoch_valid_loss, epoch_valid_acc])

    with open(os.path.join(save_path, 'log.csv'), 'w') as f:
      
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(logs)

    if epoch_valid_acc > best_valid_acc:
        best_valid_acc = epoch_valid_acc

        best_dict = {
            'epoch': epoch,
            'best_valid_acc': best_valid_acc,
            'model': model.state_dict(),
        }

        print(f"Save best model with validation accuracy: {best_valid_acc*100:.4f}%")
        torch.save(best_dict, os.path.join(save_path,'best_model.pth'))

    epoch_t = time.time() - epoch_start_t
    elapsed_time += epoch_t
    estimated_time = epoch_t * (num_epochs - epoch)

    epoch_t_gm = time.gmtime(epoch_t)
    elapsed_time_gm = time.gmtime(elapsed_time)
    estimated_time_gm = time.gmtime(estimated_time)

    print(f"- Epoch time: {epoch_t_gm.tm_hour}[h] {epoch_t_gm.tm_min}[m] {epoch_t_gm.tm_sec}[s]")
    print(f"- Elapsed time: {elapsed_time_gm.tm_hour}[h] {elapsed_time_gm.tm_min}[m] {elapsed_time_gm.tm_sec}[s]")
    print(f"- Estimated time: {estimated_time_gm.tm_hour}[h] {estimated_time_gm.tm_min}[m] {estimated_time_gm.tm_sec}[s]")

print(f"Finished with the best validation accuracy: {best_valid_acc*100:.4f}")
    
f_print.close()

