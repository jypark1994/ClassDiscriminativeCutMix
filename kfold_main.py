# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# For K-fold Cross Validation (MosquitoDL)
# - Split 'Train dataset' in k-folds.
#     - For each iteration, train with k-1 datasets, and validate with a dataset.
#     - 1 epoch = 5 fold iteration
# 

# %%
import torch
from torch import optim, nn
from torchvision import transforms, datasets
from resnet import ResNet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_folds = 5


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
    ])



    train_dataset = datasets.ImageFolder(os.path.join(root,'train'), transform=transforms_train)

    len_fold, len_fold_rest = len(train_dataset)//num_folds, len(train_dataset) % num_folds

    fold_lengths = [len_fold for x in range(num_folds)]

    if(len_fold_rest != 0):
        fold_lengths.append(len_fold_rest + len_fold)

    train_dataset = torch.utils.data.random_split(train_dataset, fold_lengths)

    train_loader = {x: torch.utils.data.DataLoader(train_dataset[x], bs_train,
                                                shuffle=True, num_workers=num_workers)
                    for x in range(num_folds)}

    test_dataset = datasets.ImageFolder(os.path.join(root,'valid'), transform=transforms_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    num_classes = 6

    return train_loader, test_loader, num_classes

num_epochs = 100

root = "../mosquitoClassification/MosquitoDL"
crop_size = 224
num_folds = 5
batch_size = (32, 32)
num_workers = 8

train_loader, test_loader, num_classes = MosquitoDL_fold(root, crop_size, num_folds, batch_size, num_workers)

# %%
class Wrapper(nn.Module):
    '''
        Author: Junyoung Park(jy_park@inu.ac.kr)
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

        def backward_hook_function(name): # Hook function for the forward pass.
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
model = ResNet('mosquitodl', 50, 6, True)
model.fc = nn.Linear(model.fc.in_features, 6)
model = nn.DataParallel(model)

pretrained_path = './pretrained/R50_ImageNet_Baseline.pth'

if pretrained_path != None:
    pretrained_dict = torch.load(pretrained_path)['state_dict']
    new_model_dict = model.state_dict()

    for k, v in new_model_dict.items():
        if 'fc' in k:
            continue
        else:
            new_model_dict[k] = pretrained_dict[k]

    model.load_state_dict(new_model_dict)
    print(f"Load pretrained state dict \'{pretrained_path}\'")

stage_names = ['layer1','layer2','layer3','layer4']

model = Wrapper(model, stage_names)

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)


# %%
from kfold_trainer import train_k_fold, test
from kfold_trainer_variants import train_k_fold_MACM, train_k_fold_MCACM
save_name = "./test.pth"


# %%
for epoch in range(num_epochs):
    print(f"==== Current Epoch: {epoch+1}")

    best_model = None
    best_test_acc = 0

    print(f"\t - Train/Val Phase ...")
    model, (epoch_train_loss, epoch_train_acc), (epoch_valid_loss, epoch_valid_acc) =         train_k_fold_MCACM(model, train_loader, optimizer, scheduler, criterion, num_folds, epoch, device,             net_type='resnet', k=1, image_priority='A', cut_prob=0, save_path='./batch_samples/', target_mode='label')

    print(f"\t - Epoch training loss : {epoch_train_loss:.4f}")
    print(f"\t - Epoch training accuracy : {epoch_train_acc*100:.4f}%")
    print(f"\t - Epoch validation loss : {epoch_valid_loss:.4f}")
    print(f"\t - Epoch validation accuracy : {epoch_valid_acc*100:.4f}%")

    print(f"\t - Test Phase ...")
    model, epoch_test_loss, epoch_test_acc = test(model, test_loader, criterion, device)
    print(f"\t - Epoch test loss : {epoch_test_loss:.4f}")
    print(f"\t - Epoch test accuracy : {epoch_test_acc*100:.4f}%")

    if epoch_test_acc > best_test_acc:
        best_test_acc = epoch_test_acc
        print(f"Save best model with test accuracy: {best_test_acc*100:.4f}%")
        torch.save(model.state_dict(), save_name)


