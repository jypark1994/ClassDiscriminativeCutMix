import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

import os
import numpy as np
import matplotlib.pyplot as plt

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

def train_k_fold(model, train_loader, optimizer, scheduler, criterion, num_folds, cur_epoch, device, **kwargs):
    """
        Author: Junyoung Park (jy_park@inu.ac.kr)

        train_k_fold - Training in K-fold cross validation.

        model(torch.nn.Module): Target model to train.
        train_loader(list(DataLoader)): Should be a list with splitted dataset.
        num_folds(int): Number of folds.
        vervose(bool): Print detailed train/val status.
    """
    epoch_train_loss = 0
    epoch_train_acc = 0
    epoch_valid_loss = 0
    epoch_valid_acc = 0

    flag_vervose = kwargs['flag_vervose']
    save_path = kwargs['save_path']


    for cur_val_fold in range(num_folds): # Iterate for each fold.
        print_v(f"\t --- Validation Fold: {cur_val_fold+1}/{num_folds}", flag_vervose)
        model.train() # Train on training folds.

        mean_fold_train_loss = 0
        mean_fold_train_acc = 0

        for cur_train_fold in range(num_folds):

            fold_train_loss = 0
            fold_train_n_corrects = 0
            fold_train_n_samples = 0

            if(cur_train_fold == cur_val_fold):
                print_v(f"\t\t - Skipping validation fold ({cur_val_fold+1}/{num_folds})", flag_vervose)
                continue

            print_v(f"\t\t - Training: {cur_train_fold+1}/{num_folds}", flag_vervose)

            for idx, data in enumerate(train_loader[cur_train_fold]):

                batch, labels = data[0].to(device), data[1].to(device)
                
                optimizer.zero_grad()

                pred = model(batch)
                pred_max = torch.argmax(pred, 1)

                loss = criterion(pred, labels)

                fold_train_loss += loss.detach().cpu().numpy()
                fold_train_n_samples += labels.size(0)
                fold_train_n_corrects += torch.sum(pred_max == labels).detach().cpu().numpy()

                if idx == len(train_loader[cur_train_fold])//2 and cur_epoch % 1 == 0:
                    input_ex = make_grid(batch.detach().cpu(), normalize=True, nrow=8, padding=2).permute([1,2,0])
                    fig, ax = plt.subplots(1,1,figsize=(8,4))
                    ax.imshow(input_ex)
                    ax.set_title(f"TrainVal Batch Examples")
                    ax.axis('off')
                    fig.savefig(os.path.join(save_path, f"TrainVal_BatchSample_E{cur_epoch}_F{idx}.png"))
                    plt.draw()
                    plt.clf()
                    plt.close("all")

                loss.backward()
                optimizer.step()
                scheduler.step()

            fold_train_loss /= fold_train_n_samples
            fold_train_acc = fold_train_n_corrects / fold_train_n_samples

            mean_fold_train_loss += fold_train_loss
            mean_fold_train_acc += fold_train_acc    

            print_v(f"\t\t\t - Fold training loss : {fold_train_loss:.4f}", flag_vervose)
            print_v(f"\t\t\t - Fold training accuracy : {fold_train_acc*100:.4f}%", flag_vervose)

        mean_fold_train_loss /= (num_folds - 1)
        mean_fold_train_acc /= (num_folds - 1)
        print_v(f"\t\t - Mean Fold training loss : {mean_fold_train_loss:.4f}", flag_vervose)
        print_v(f"\t\t - Mean Fold training accuracy : {mean_fold_train_acc*100:.4f}%", flag_vervose)

        fold_valid_loss = 0
        fold_valid_n_corrects = 0
        fold_valid_n_samples = 0

        model.eval()
        with torch.no_grad():
            print_v(f"\t\t - Validation: {cur_val_fold+1}/{num_folds}", flag_vervose)
            for idx, batch in enumerate(train_loader[cur_val_fold]):
                batch, labels = data[0].to(device), data[1].to(device)
 
                pred = model(batch)
                pred_max = torch.argmax(pred, 1)

                loss = criterion(pred, labels)

                fold_valid_loss += loss.detach().cpu().numpy()
                fold_valid_n_samples += labels.size(0)
                fold_valid_n_corrects += torch.sum(pred_max == labels).detach().cpu().numpy()

        fold_valid_loss /= fold_valid_n_samples
        fold_valid_acc = fold_valid_n_corrects / fold_valid_n_samples

        print_v(f"\t\t\t - Fold validation loss : {fold_valid_loss:.4f}", flag_vervose)
        print_v(f"\t\t\t - Fold validation accuracy : {fold_valid_acc*100:.4f}%", flag_vervose)

        epoch_train_loss += fold_train_loss
        epoch_train_acc += fold_train_acc
        epoch_valid_loss += fold_valid_loss
        epoch_valid_acc += fold_valid_acc
    
    epoch_train_loss /= num_folds
    epoch_train_acc /= num_folds
    epoch_valid_loss /= num_folds
    epoch_valid_acc /= num_folds

    return model, (epoch_train_loss, epoch_train_acc), (epoch_valid_loss, epoch_valid_acc)

def train_k_fold_MACM(model, train_loader, optimizer, scheduler, criterion, num_folds, cur_epoch, device, **kwargs):
    """
        Author: Junyoung Park (jy_park@inu.ac.kr)

        train_k_fold_MACM - Training in K-fold cross validation.

        model(torch.nn.Module): Target model to train.
        train_loader(list(DataLoader)): Should be a list with splitted dataset.
        num_folds(int): Number of folds (k).
        vervose(bool): Print detailed train/val status.
    """
    epoch_train_loss = 0
    epoch_train_acc = 0
    epoch_valid_loss = 0
    epoch_valid_acc = 0

    k = kwargs['k']
    image_priority = kwargs['image_priority']
    cut_prob = kwargs['cut_prob']
    flag_vervose = kwargs['flag_vervose']
    save_path = kwargs['save_path']

    for cur_val_fold in range(num_folds): # Iterate for each fold.
        print_v(f"\t --- Validation Fold: {cur_val_fold+1}/{num_folds}", flag_vervose)
        model.train() # Train on training folds.

        mean_fold_train_loss = 0
        mean_fold_train_acc = 0

        for cur_train_fold in range(num_folds):

            fold_train_loss = 0
            fold_train_n_corrects = 0
            fold_train_n_samples = 0

            if(cur_train_fold == cur_val_fold):
                print_v(f"\t\t - Skipping validation fold ({cur_val_fold+1}/{num_folds})", flag_vervose)
                continue

            print_v(f"\t\t - Training: {cur_train_fold+1}/{num_folds}", flag_vervose)

            for idx, data in enumerate(train_loader[cur_train_fold]):

                batch, labels = data[0].to(device), data[1].to(device)
                
                optimizer.zero_grad()

                pred = model(batch)
                pred_max = torch.argmax(pred, 1)

                target_stage_name = 'None'

                r = np.random.rand(1)

                top_k_for_stage = 0

                if r < cut_prob:
                    target_stage_index = torch.randint(low=0, high=model.num_stages, size=(1,))[0]
                    target_stage_name = model.stage_names[target_stage_index]
                    target_fmap = model.dict_activation[target_stage_name].mean(dim=1)

                    model.clear_dict()

                    N, W_f, H_f = target_fmap.shape

                    top_k_for_stage = k * (4**(model.num_stages - target_stage_index - 1))
                    
                    attention_masks = generate_attentive_mask(target_fmap, top_k = top_k_for_stage)

                    upsampled_attention_masks = F.interpolate(attention_masks.unsqueeze(1).repeat([1,3,1,1]), 
                        size=batch.shape[-2:], mode='nearest')

                    n_occluded_pixels = top_k_for_stage
                    n_total_pixels = W_f * H_f

                    rand_index = torch.randperm(batch.size()[0]).cuda()

                    if image_priority == 'A':
                        image_a = (1 - upsampled_attention_masks) * batch
                        image_b = upsampled_attention_masks * batch[rand_index]
                    elif image_priority == 'B':
                        image_a = upsampled_attention_masks * batch
                        image_b = (1-upsampled_attention_masks) * batch[rand_index]

                    batch = image_a + image_b
                    
                    occlusion_ratio = (n_occluded_pixels/n_total_pixels) # 1 - Mixed Ratio

                    target_a = labels
                    target_b = labels[rand_index]

                    optimizer.zero_grad()

                    pred = model(batch)

                    loss = criterion(pred, target_a)  * occlusion_ratio + criterion(pred, target_b) * (1 - occlusion_ratio)

                else:
                    loss = criterion(pred, labels)

                if idx == len(train_loader[cur_train_fold])//2 and cur_epoch % 1 == 0:
                    input_ex = make_grid(batch.detach().cpu(), normalize=True, nrow=8, padding=2).permute([1,2,0])
                    fig, ax = plt.subplots(1,1,figsize=(8,4))
                    ax.imshow(input_ex)
                    ax.set_title(f"TrainVal MACM Batch Examples\nCut_Prob:{cut_prob}, Cur_Target: {target_stage_name}, Num_occlusion: {top_k_for_stage} ")
                    ax.axis('off')
                    fig.savefig(os.path.join(save_path, f"TrainVal_BatchSample_E{cur_epoch}_F{cur_train_fold}_I{idx}.png"))
                    plt.draw()
                    plt.clf()
                    plt.close("all")

                fold_train_loss += loss.detach().cpu().numpy()
                fold_train_n_samples += labels.size(0)
                fold_train_n_corrects += torch.sum(pred_max == labels).detach().cpu().numpy()

                loss.backward()
                optimizer.step()
                scheduler.step()

            fold_train_loss /= fold_train_n_samples
            fold_train_acc = fold_train_n_corrects / fold_train_n_samples

            mean_fold_train_loss += fold_train_loss
            mean_fold_train_acc += fold_train_acc    

            print_v(f"\t\t\t - Fold training loss : {fold_train_loss:.4f}", flag_vervose)
            print_v(f"\t\t\t - Fold training accuracy : {fold_train_acc*100:.4f}%", flag_vervose)

        mean_fold_train_loss /= (num_folds - 1)
        mean_fold_train_acc /= (num_folds - 1)
        print_v(f"\t\t - Mean Fold training loss : {mean_fold_train_loss:.4f}", flag_vervose)
        print_v(f"\t\t - Mean Fold training accuracy : {mean_fold_train_acc*100:.4f}%", flag_vervose)

        fold_valid_loss = 0
        fold_valid_n_corrects = 0
        fold_valid_n_samples = 0

        model.eval()
        with torch.no_grad():
            print_v(f"\t\t - Validation: {cur_val_fold+1}/{num_folds}", flag_vervose)
            for idx, batch in enumerate(train_loader[cur_val_fold]):
                batch, labels = data[0].to(device), data[1].to(device)
 
                pred = model(batch)
                pred_max = torch.argmax(pred, 1)

                loss = criterion(pred, labels)

                fold_valid_loss += loss.detach().cpu().numpy()            
                fold_valid_n_samples += labels.size(0)
                fold_valid_n_corrects += torch.sum(pred_max == labels).detach().cpu().numpy()            

        fold_valid_loss /= fold_valid_n_samples
        fold_valid_acc = fold_valid_n_corrects / fold_valid_n_samples

        print_v(f"\t\t\t - Fold validation loss : {fold_valid_loss:.4f}", flag_vervose)
        print_v(f"\t\t\t - Fold validation accuracy : {fold_valid_acc*100:.4f}%", flag_vervose)

        epoch_train_loss += fold_train_loss
        epoch_train_acc += fold_train_acc
        epoch_valid_loss += fold_valid_loss
        epoch_valid_acc += fold_valid_acc
    
    epoch_train_loss /= num_folds
    epoch_train_acc /= num_folds
    epoch_valid_loss /= num_folds
    epoch_valid_acc /= num_folds

    return model, (epoch_train_loss, epoch_train_acc), (epoch_valid_loss, epoch_valid_acc)

def train_k_fold_MCACM(model, train_loader, optimizer, scheduler, criterion, num_folds, cur_epoch, device, **kwargs):
    """
        Author: Junyoung Park (jy_park@inu.ac.kr)

        train_k_fold_MCACM - Training in K-fold cross validation.

        model(torch.nn.Module): Target model to train.
        train_loader(list(DataLoader)): Should be a list with splitted dataset.
        num_folds(int): Number of folds (k).
        vervose(bool): Print detailed train/val status.
    """
    epoch_train_loss = 0
    epoch_train_acc = 0
    epoch_valid_loss = 0
    epoch_valid_acc = 0

    k = kwargs['k']
    image_priority = kwargs['image_priority']
    cut_prob = kwargs['cut_prob']
    cam_mode = kwargs['cam_mode']
    flag_vervose = kwargs['flag_vervose']
    save_path = kwargs['save_path']

    for cur_val_fold in range(num_folds): # Iterate for each fold.
        print_v(f"\t --- Validation Fold: {cur_val_fold+1}/{num_folds}", flag_vervose)
        model.train() # Train on training folds.

        mean_fold_train_loss = 0
        mean_fold_train_acc = 0

        for cur_train_fold in range(num_folds):

            fold_train_loss = 0
            fold_train_n_corrects = 0
            fold_train_n_samples = 0

            if(cur_train_fold == cur_val_fold):
                print_v(f"\t\t - Skipping validation fold ({cur_val_fold+1}/{num_folds})", flag_vervose)
                continue

            print_v(f"\t\t - Training: {cur_train_fold+1}/{num_folds}", flag_vervose)

            for idx, data in enumerate(train_loader[cur_train_fold]):

                batch, labels = data[0].to(device), data[1].to(device)
                
                optimizer.zero_grad()

                pred = model(batch)
                pred_max = torch.argmax(pred, 1)

                target_stage_name = 'None'

                r = np.random.rand(1)

                top_k_for_stage = 0

                if r < cut_prob:
                    target_stage_index = torch.randint(low=0, high=model.num_stages, size=(1,))[0]
                    target_stage_name = model.stage_names[target_stage_index]

                    if cam_mode == 'label':
                        # Mode 1 : Using target(Label) loss for generating CAMs
                        loss = criterion(pred, labels)
                    
                    elif cam_mode == 'likely':
                        # Mode 2 : Using "Most Likely Class" loss for generating CAMs
                        most_confident_target = pred_max
                        loss = criterion(pred, most_confident_target)
                    else:
                        assert "Target mode is not specified. (Use \'label\' or \'likely\')"

                    loss.backward()

                    target_fmap = model.dict_activation[target_stage_name]
                    target_gradients = model.dict_gradients[target_stage_name][0]
                    
                    model.clear_dict()

                    N, C, W_f, H_f = target_fmap.shape

                    importance_weights = F.adaptive_avg_pool2d(target_gradients, 1) # [N x C x 1 x 1]

                    class_activation_map = torch.mul(target_fmap, importance_weights).sum(dim=1, keepdim=True) # [N x 1 x W_f x H_f]
                    class_activation_map = F.relu(class_activation_map).squeeze(dim=1) # [N x W_f x H_f]

                    top_k_for_stage = k * (4**(model.num_stages - target_stage_index - 1))
                    
                    attention_masks = generate_attentive_mask(class_activation_map, top_k = top_k_for_stage)

                    upsampled_attention_masks = F.interpolate(attention_masks.unsqueeze(1).repeat([1,3,1,1]), 
                        size=batch.shape[-2:], mode='nearest')

                    n_occluded_pixels = top_k_for_stage
                    n_total_pixels = W_f * H_f

                    rand_index = torch.randperm(batch.size()[0]).cuda()

                    if image_priority == 'A':
                        image_a = (1 - upsampled_attention_masks) * batch
                        image_b = upsampled_attention_masks * batch[rand_index]
                    elif image_priority == 'B':
                        image_a = upsampled_attention_masks * batch
                        image_b = (1-upsampled_attention_masks) * batch[rand_index]

                    batch = image_a + image_b
                    
                    occlusion_ratio = (n_occluded_pixels/n_total_pixels) # 1 - Mixed Ratio

                    target_a = labels
                    target_b = labels[rand_index]

                    optimizer.zero_grad()

                    pred = model(batch)
                    pred_max = torch.argmax(pred, 1)

                    loss = criterion(pred, target_a)  * occlusion_ratio + criterion(pred, target_b) * (1 - occlusion_ratio)

                else:
                    loss = criterion(pred, labels)

                if idx == len(train_loader[cur_train_fold])//2 and cur_epoch % 1 == 0:
                    input_ex = make_grid(batch.detach().cpu(), normalize=True, nrow=8, padding=2).permute([1,2,0])
                    fig, ax = plt.subplots(1,1,figsize=(8,4))
                    ax.imshow(input_ex)
                    ax.set_title(f"TrainVal MCACM Batch Examples\nCut_Prob:{cut_prob}, Cur_Target: {target_stage_name}, Num_occlusion: {top_k_for_stage} ")
                    ax.axis('off')
                    fig.savefig(os.path.join(save_path, f"TrainVal_BatchSample_E{cur_epoch}_F{cur_train_fold}_I{idx}.png"))
                    plt.draw()
                    plt.clf()
                    plt.close("all")
                fold_train_loss += loss.detach().cpu().numpy()
                fold_train_n_samples += labels.size(0)
                fold_train_n_corrects += torch.sum(pred_max == labels).detach().cpu().numpy()

                loss.backward()
                optimizer.step()
                scheduler.step()

            fold_train_loss /= fold_train_n_samples
            fold_train_acc = fold_train_n_corrects / fold_train_n_samples

            mean_fold_train_loss += fold_train_loss
            mean_fold_train_acc += fold_train_acc    

            print_v(f"\t\t\t - Fold training loss : {fold_train_loss:.4f}", flag_vervose)
            print_v(f"\t\t\t - Fold training accuracy : {fold_train_acc*100:.4f}%", flag_vervose)

        mean_fold_train_loss /= (num_folds - 1)
        mean_fold_train_acc /= (num_folds - 1)
        print_v(f"\t\t - Mean Fold training loss : {mean_fold_train_loss:.4f}", flag_vervose)
        print_v(f"\t\t - Mean Fold training accuracy : {mean_fold_train_acc*100:.4f}%", flag_vervose)

        fold_valid_loss = 0
        fold_valid_n_corrects = 0
        fold_valid_n_samples = 0

        model.eval()
        with torch.no_grad():
            print_v(f"\t\t - Validation: {cur_val_fold+1}/{num_folds}", flag_vervose)
            for idx, batch in enumerate(train_loader[cur_val_fold]):
                batch, labels = data[0].to(device), data[1].to(device)
 
                pred = model(batch)
                pred_max = torch.argmax(pred, 1)

                loss = criterion(pred, labels)

                fold_valid_loss += loss.detach().cpu().numpy()            
                fold_valid_n_samples += labels.size(0)
                fold_valid_n_corrects += torch.sum(pred_max == labels).detach().cpu().numpy()            

        fold_valid_loss /= fold_valid_n_samples
        fold_valid_acc = fold_valid_n_corrects / fold_valid_n_samples

        print_v(f"\t\t\t - Fold validation loss : {fold_valid_loss:.4f}", flag_vervose)
        print_v(f"\t\t\t - Fold validation accuracy : {fold_valid_acc*100:.4f}%", flag_vervose)

        epoch_train_loss += fold_train_loss
        epoch_train_acc += fold_train_acc
        epoch_valid_loss += fold_valid_loss
        epoch_valid_acc += fold_valid_acc
    
    epoch_train_loss /= num_folds
    epoch_train_acc /= num_folds
    epoch_valid_loss /= num_folds
    epoch_valid_acc /= num_folds

    return model, (epoch_train_loss, epoch_train_acc), (epoch_valid_loss, epoch_valid_acc)

def test(model, test_loader, criterion, device, save_path, cur_epoch):

    test_loss = 0
    test_acc = 0

    test_n_samples = 0
    test_n_corrects = 0

    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            batch, labels = data[0].to(device), data[1].to(device)

            pred = model(batch)
            pred_max = torch.argmax(pred, 1)

            loss = criterion(pred, labels)

            test_loss += loss.detach().cpu().numpy()
            test_n_samples += labels.size(0)
            test_n_corrects += torch.sum(pred_max == labels).detach().cpu().numpy()


            if idx%100 == 0 and cur_epoch % 20 == 0:
                input_ex = make_grid(batch.detach().cpu(), normalize=True, nrow=8, padding=2).permute([1,2,0])
                fig, ax = plt.subplots(1,1,figsize=(8,4))
                ax.imshow(input_ex)
                ax.set_title(f"Testing Batch Examples")
                ax.axis('off')
            
                fig.savefig(os.path.join(save_path, f"Test_BatchSample_E{cur_epoch}_I{idx}.png"))
                plt.draw()
                plt.clf()
                plt.close("all")

    test_loss /= test_n_samples
    test_acc = test_n_corrects/test_n_samples

    return model, test_loss, test_acc