import torch
from torch import optim, nn
from torchvision import transforms, datasets, models

import os
import sys
import csv
import time
import random
import numpy as np
import argparse

from kfold_trainer import *
from utils import Wrapper
from datasets import *

parser = argparse.ArgumentParser(description='Train and Evaluate MosquitoDL using k-fold validation')
parser.add_argument('--net_type', default='resnet50', type=str,
                    help='networktype: resnet')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--gpus', type=str, default='0')

parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--num_folds', type=int, default=5)

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=5e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--scheduler_step', type=int, default=25)

parser.add_argument('--expr_name', type=str, default="default")
parser.add_argument('--dataset_root', type=str, default="~/datasets/MosquitoDL_TestAug")
parser.add_argument('--flag_vervose', action='store_true')
parser.add_argument('--single_scale', action='store_true')
parser.add_argument('--deterministic', action='store_true')

parser.add_argument('--train_mode', type=str, default="cutmix")
parser.add_argument('--cut_mode', type=str, default="A")
parser.add_argument('--cam_mode', type=str, default="label")
parser.add_argument('--mask_mode', type=str, default="top")
parser.add_argument('--k', type=int, default=1)
parser.add_argument('--threshold', type=float, default=0.1)
parser.add_argument('--cut_prob', type=float, default=0.5)

args = parser.parse_args()

# ---- Randomness Control ----
if args.deterministic:
    rand_seed = 0

    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ----------------------------

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_root = args.dataset_root
save_path = os.path.join("./results", args.expr_name)

crop_size = 224 # Default

net_type = args.net_type.lower()
train_mode = args.train_mode

num_epochs = args.num_epochs
num_folds = args.num_folds
batch_size = (args.batch_size, args.batch_size)
num_workers = args.num_workers
flag_vervose = args.flag_vervose
single_scale = args.single_scale
scheduler_step = args.scheduler_step
target_mode = args.cam_mode
mask_mode = args.mask_mode
threshold = args.threshold

os.makedirs(save_path, exist_ok=True)

f_print = open(os.path.join(save_path, 'output.txt'), 'w')

sys.stdout = f_print # Change the standard output to the file we created.

print(args)



train_loader, test_loader, num_classes = MosquitoDL_fold(dataset_root, crop_size, num_folds, batch_size, num_workers, ver='v1')

if net_type == 'resnet50':
    model = models.resnet50(pretrained=args.pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = nn.DataParallel(model)
elif net_type == 'resnet18':
    model = models.resnet18(pretrained=args.pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = nn.DataParallel(model)
elif net_type == 'mobilenetv2':
    model = models.mobilenet_v2(pretrained=args.pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = nn.DataParallel(model)
elif net_type == 'vgg16':
    model = models.vgg16(pretrained=args.pretrained)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model = nn.DataParallel(model)
else:
    assert "Invalid 'net_type' !"

# pretrained_path = './pretrained/R50_ImageNet_Baseline.pth'

# if pretrained_path != None:
#     pretrained_dict = torch.load(pretrained_path)['state_dict']
#     new_model_dict = model.state_dict()

#     for k, v in new_model_dict.items():
#         if 'fc' in k:
#             continue
#         else:
#             new_model_dict[k] = pretrained_dict[k]

#     model.load_state_dict(new_model_dict)
#     print(f"Load pretrained state dict \'{pretrained_path}\'")

if 'resnet' in net_type:
    if single_scale:
        stage_names = ['layer4']
    else:
        stage_names = ['layer1','layer2','layer3','layer4']
elif net_type == 'mobilenetv2':
    if single_scale:
        stage_names = ['features.14']
    else:
        stage_names = ['features.2','features.4','features.7','features.14']
elif net_type == 'vgg16':
    if single_scale:
        stage_names = ['features.23']
    else:
        stage_names = ['features.4', 'features.9', 'features.16','features.24']
else:
    assert "Unsupported network type !"


model = Wrapper(model, stage_names) # Wrapper for registering hooks for 'stage_names' of the 'model'.

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.75)

# %%

logs = []
logs.append(['epoch', 'loss_tr', 'acc_tr', 'loss_val', 'acc_val', 'loss_test', 'acc_test'])
elapsed_time = 0
best_model = None
best_test_acc = 0

for epoch in range(num_epochs):
    print(f"==== Current Epoch: {epoch+1}")


    epoch_start_t = time.time()

    print(f"\t - Train/Val Phase ...")

    if train_mode == 'vanilla':
        model, (epoch_train_loss, epoch_train_acc), (epoch_valid_loss, epoch_valid_acc) = \
            train_k_fold(model, train_loader, optimizer, scheduler, criterion, num_folds, epoch, device, flag_vervose=flag_vervose, \
            net_type='resnet', save_path=save_path)
    elif train_mode == 'cutmix':
        model, (epoch_train_loss, epoch_train_acc), (epoch_valid_loss, epoch_valid_acc) = \
            train_k_fold_CutMix(model, train_loader, optimizer, scheduler, criterion, num_folds, epoch, device, flag_vervose=flag_vervose, \
            net_type='resnet', cut_prob=args.cut_prob, save_path=save_path)
    elif train_mode == 'MACM': # Multiscale Attentive Cutmix
        model, (epoch_train_loss, epoch_train_acc), (epoch_valid_loss, epoch_valid_acc) = \
            train_k_fold_MACM(model, train_loader, optimizer, scheduler, criterion, num_folds, epoch, device, flag_vervose=flag_vervose, \
            net_type='resnet', k=args.k, image_priority=args.cut_mode, cut_prob=args.cut_prob, save_path=save_path, target_mode=args.cam_mode, \
            mask_mode=mask_mode, threshold=threshold)
    elif train_mode == 'MCACM': # Multiscale Class Attentive Cutmix
        model, (epoch_train_loss, epoch_train_acc), (epoch_valid_loss, epoch_valid_acc) = \
            train_k_fold_MCACM(model, train_loader, optimizer, scheduler, criterion, num_folds, epoch, device, flag_vervose=flag_vervose, \
            net_type='resnet', k=args.k, image_priority=args.cut_mode, cut_prob=args.cut_prob, cam_mode=args.cam_mode, save_path=save_path, target_mode=args.cam_mode, \
            mask_mode=mask_mode, threshold=threshold)
    else:
        assert 'Invalid training mode !'

    print(f"\t - Epoch training loss : {epoch_train_loss:.4f}")
    print(f"\t - Epoch training accuracy : {epoch_train_acc*100:.4f}%")
    print(f"\t - Epoch validation loss : {epoch_valid_loss:.4f}")
    print(f"\t - Epoch validation accuracy : {epoch_valid_acc*100:.4f}%")

    print(f"\t - Test Phase ...")
    model, epoch_test_loss, epoch_test_acc = test(model, test_loader, criterion, device, save_path, epoch)
    print(f"\t - Epoch test loss : {epoch_test_loss:.4f}")
    print(f"\t - Epoch test accuracy : {epoch_test_acc*100:.4f}%")

    logs.append([epoch, epoch_train_loss, epoch_train_acc, \
        epoch_valid_loss, epoch_valid_acc, \
        epoch_test_loss, epoch_test_acc])

    with open(os.path.join(save_path, 'log.csv'), 'w') as f:
      
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(logs)

    if epoch_test_acc > best_test_acc:
        best_test_acc = epoch_test_acc

        best_dict = {
            'epoch': epoch,
            'best_test_acc': best_test_acc,
            'model': model.state_dict(),
        }

        print(f"Save best model with test accuracy: {best_test_acc*100:.4f}%")
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

print(f"Finished with the best test accuracy: {best_test_acc*100:.4f}")

f_print.close()

