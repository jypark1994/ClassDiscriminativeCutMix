import torch

def train_k_fold(model, train_loader, optimizer, scheduler, criterion, num_folds, cur_epoch, device, **kwargs):
    """
        Author: Junyoung Park (jy_park@inu.ac.kr)

        train_k_fold - Training in K-fold cross validation.

        model(torch.nn.Module): Target model to train.
        train_loader(list(DataLoader)): Should be a list with splitted dataset.
        num_folds(int): Number of folds (k).
    """
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
                    
                    # CutMix Variants will come here ...
                    

                    optimizer.zero_grad()

                    pred = model(batch)
                    pred_max = torch.argmax(pred, 1)

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


def test(model, test_loader, criterion, device):

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

            test_loss += loss
            test_n_samples += labels.size(0)
            test_n_corrects += torch.sum(pred_max == labels)

    test_loss /= test_n_samples
    test_acc = test_n_corrects/test_n_samples

    return model, test_loss, test_acc