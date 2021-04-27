import torch
import torch.nn.functional as F
import numpy as np

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
    top_indices = indices[:, :top_k] # [N, Top_k]

    mask = torch.ones_like(x)

    for i in range(N):
        mask[i, top_indices[i]] = 0
    
    mask = mask.reshape([N, W, H])
    # print(mask)

    return mask

def train_k_fold_MCACM(model, train_loader, optimizer, scheduler, criterion, num_folds, cur_epoch, device, **kwargs):
    """
        Author: Junyoung Park (jy_park@inu.ac.kr)

        train_k_fold_MCACM - Training Multiscale Class Activation CutMix using K-fold cross validation.

        model(torch.nn.Module): Target model to train.
        train_loader(list(DataLoader)): Should be a list with splitted dataset.
        num_folds(int): Number of folds (k).

    """
    k = kwargs['k']
    image_priority = kwargs['image_priority']
    cut_prob = kwargs['cut_prob']

    epoch_train_loss = 0
    epoch_train_acc = 0
    epoch_valid_loss = 0
    epoch_valid_acc = 0

    for cur_val_fold in range(num_folds): # Iterate for each fold.
        # print(f"\t - Fold: {cur_val_fold+1}/{num_folds}")

        fold_train_loss = 0
        fold_train_n_corrects = 0
        fold_train_n_samples = 0

        model.train() # Train on training folds.
        for cur_train_fold in range(num_folds):
            if(cur_train_fold != cur_val_fold):
                # print(f"\t\t - Training: {cur_train_fold}")
                for idx, data in enumerate(train_loader[cur_train_fold]):

                    batch, labels = data[0].to(device), data[1].to(device)

                    optimizer.zero_grad()

                    pred = model(batch)
                    pred_max = torch.argmax(pred, 1)

                    # CutMix Variants will come here ...

                    target_stage_name = 'None'

                    r = np.random.rand(1)

                    if r < cut_prob:

                        target_stage_index = torch.randint(low=0, high=model.num_stages, size=(1,))[0]
                        target_stage_name = model.stage_names[target_stage_index]
                        target_fmap = model.dict_activation[target_stage_name].mean(dim=1)
 
                        model.clear_dict()

                        N, W_f, H_f = target_fmap.shape

                        top_k_for_stage = k * (4**(model.num_stages - target_stage_index))
                        
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

                        output = model(batch)

                        loss = criterion(output, target_a)  * occlusion_ratio + criterion(output, target_b) * (1 - occlusion_ratio)

                    else:

                        loss = criterion(pred, labels)

                    fold_train_loss += loss
                    fold_train_n_samples += labels.size(0)
                    fold_train_n_corrects += torch.sum(pred_max == labels)
                    

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            
            else:
                continue # Ignore a validation fold.

        fold_valid_loss = 0
        fold_valid_n_corrects = 0
        fold_valid_n_samples = 0


        model.eval()
        with torch.no_grad():
            # print(f"\t\t - Validation: {cur_val_fold}")
            for idx, batch in enumerate(train_loader[cur_val_fold]):
                batch, labels = data[0].to(device), data[1].to(device)
 
                pred = model(batch)
                pred_max = torch.argmax(pred, 1)

                loss = criterion(pred, labels)

                fold_valid_loss += loss
                fold_valid_n_samples += labels.size(0)
                fold_valid_n_corrects += torch.sum(pred_max == labels)

        fold_train_loss /= fold_train_n_samples
        fold_train_acc = fold_train_n_corrects / fold_train_n_samples                

        fold_valid_loss /= fold_valid_n_samples
        fold_valid_acc = fold_valid_n_corrects / fold_valid_n_samples

        # print(f"\t\t - Fold training loss : {fold_train_loss:.4f}")
        # print(f"\t\t - Fold training accuracy : {fold_train_acc*100:.4f}%")
        # print(f"\t\t - Fold validation loss : {fold_valid_loss:.4f}")
        # print(f"\t\t - Fold validation accuracy : {fold_valid_acc*100:.4f}%")

        epoch_train_loss += fold_train_loss
        epoch_train_acc += fold_train_acc
        epoch_valid_loss += fold_valid_loss
        epoch_valid_acc += fold_valid_acc
    
    epoch_train_loss /= num_folds
    epoch_train_acc /= num_folds
    epoch_valid_loss /= num_folds
    epoch_valid_acc /= num_folds

    return model, (epoch_train_loss, epoch_train_acc), (epoch_valid_loss, epoch_valid_acc)