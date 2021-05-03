import torch
from torch import optim, nn
from torchvision import transforms, datasets, models

import os
import sys
import csv
import time
import argparse

from kfold_trainer import train_k_fold, train_k_fold_MACM, train_k_fold_MCACM, test

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
parser.add_argument('--dataset_root', type=str, default="~/datasets/MosquitoDL")
parser.add_argument('--flag_vervose', action='store_true')
parser.add_argument('--single_scale', action='store_true')

parser.add_argument('--train_mode', type=str, default="cutmix")
parser.add_argument('--cut_mode', type=str, default="A")
parser.add_argument('--cam_mode', type=str, default="label")
parser.add_argument('--k', type=int, default=1)
parser.add_argument('--cut_prob', type=float, default=0.5)

args = parser.parse_args()

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

os.makedirs(save_path, exist_ok=True)

f_print = open(os.path.join(save_path, 'output.txt'), 'w')

sys.stdout = f_print # Change the standard output to the file we created.

print(args)

def MosquitoDL_fold(root, crop_size=224, num_folds=5, batch_size=(64, 32), num_workers=8):
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

train_loader, test_loader, num_classes = MosquitoDL_fold(dataset_root, crop_size, num_folds, batch_size, num_workers)

# %%
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


# %%
model = models.resnet50(pretrained=args.pretrained)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = nn.DataParallel(model)

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
    stage_names = ['layer1','layer2','layer3','layer4']
elif net_type == 'mobilenetv2':
    stage_names = ['features.2','features.4','features.7','features.14']
else:
    assert "Unsupported network type !"

if single_scale:
        stage_names = stage_names[-1]

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
            net_type='resnet', k=args.k, image_priority=args.cut_mode, cut_prob=args.cut_prob, save_path=save_path, target_mode='label')
    elif train_mode == 'MCACM': # Multiscale Class Attentive Cutmix
        model, (epoch_train_loss, epoch_train_acc), (epoch_valid_loss, epoch_valid_acc) = \
            train_k_fold_MCACM(model, train_loader, optimizer, scheduler, criterion, num_folds, epoch, device, flag_vervose=flag_vervose, \
            net_type='resnet', k=args.k, image_priority=args.cut_mode, cut_prob=args.cut_prob, cam_mode=args.cam_mode, save_path=save_path, target_mode='label')
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

