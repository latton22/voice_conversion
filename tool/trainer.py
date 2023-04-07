import torch
import optuna
import json
import csv
from tqdm import tqdm

from dataset import VCDataset
from model import ContextNet, SpecAugment
from lr_scheduler import Transformer_LRScheduler
from rich.console import Console
from rich.traceback import install
from rich.progress import track

import os
import sys
from importlib import import_module
config = sys.argv[1]
config_dir = os.path.dirname(config)
config_bname = os.path.splitext(os.path.basename(config))[0]
sys.path.append(config_dir)
config = import_module(config_bname)

def updater(train_loader, valid_loader, augment, model, optimizer, scheduler, criterion, device):
    train_loss = []
    val_loss = []
    model.train()
    for data in track(train_loader, total=len(train_loader), description='Training model...'):
        inputs, outputs = data
        repeated_inputs = inputs.repeat(config.batchsize, 1, 1)
        repeated_outputs = outputs.repeat(config.batchsize, 1)
        """
        inputs = inputs.squeeze()
        outputs = outputs.squeeze()
        repeated_inputs = torch.stack([inputs for _ in range(config.batchsize)])
        repeated_outputs = torch.stack([outputs for _ in range(config.batchsize)])
        """
        repeated_inputs = repeated_inputs.to(device, dtype=torch.float)
        repeated_outputs = repeated_outputs.to(device, dtype=torch.float)

        optimizer.zero_grad()

        augmented_inputs, augmented_outputs = augment(repeated_inputs, repeated_outputs)
        converted = model(augmented_inputs)
        converted = converted.permute(0,2,1)
        loss = criterion(converted, augmented_outputs.long())
        train_loss.append(loss.item())
        print('train loss: {}, lr: {}'.format(loss, scheduler.get_last_lr()[0]))

        loss.backward()
        del loss
        optimizer.step()
        scheduler.step()
    val_loss.append(validation(valid_loader, model, optimizer, criterion, device))
    with open(config.valid_path, mode='a') as f:
        f.write('last, {}\n'.format(val_loss[-1]))
    return train_loss, val_loss

def validation(valid_loader, model, optimizer, criterion, device):
    val_loss   = 0.0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            inputs, outputs = data
            inputs = inputs.to(device, dtype=torch.float)
            outputs = outputs.to(device, dtype=torch.float)
            converted = model(inputs)
            converted = converted.squeeze()
            outputs = outputs.squeeze()
            loss = criterion(converted, outputs.long())
            total += outputs.size(0)
            val_loss += loss.item() * outputs.size(0)
    val_loss /= float(total)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('validation loss: {}'.format(val_loss))
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    return val_loss

def main():
    train_set     = VCDataset(config.src_train_dir, config.tgt_train_dir, config.mspec_dim, config.ppg_dim)
    train_loader  = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
    valid_set     = VCDataset(config.src_valid_dir, config.tgt_valid_dir, config.mspec_dim, config.ppg_dim)
    valid_loader  = torch.utils.data.DataLoader(valid_set , batch_size=1, shuffle=True, num_workers=2)

    gpu_id = sys.argv[2]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:"+ str(gpu_id) if use_cuda else "cpu")
    print(device)

    with open(config.train_path, mode='w') as f:
        f.write('iterator, train_loss\n')
    with open(config.valid_path, mode='w') as f:
        f.write('iterator, validation_loss\n')
    """
    with open(config.lr_log_path, mode='w') as f:
        f.write('iterator, lr\n')
    """

    #setting model
    augment = SpecAugment(freq_mask_num=1, time_mask_num=1, freq_width=27//2, maximum_time_mask_ratio=0.05)
    augment = augment.to(device)
    model = ContextNet()
    model = model.to(device)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-6)
    scheduler = Transformer_LRScheduler(optimizer, warmup_steps=config.warmup_steps)

    # train
    for e in range(config.n_epoch):
        train_loss, val_loss = updater(train_loader, valid_loader, augment, model, optimizer, scheduler, criterion, device)
        print('******************************************************************')
        print("train: {}, validation: {}".format(train_loss[-1], val_loss[-1]))
        print('******************************************************************')
        if e == 0:
            print("save best model")
            pred = val_loss[-1]
            with open(config.train_path, mode='a') as f:
                for i in range(len(train_loss)):
                    f.write('{}, {}\n'.format(i+1, train_loss[i]))
            torch.save(model.state_dict(), config.model_path.replace('epoch', 'best'))
        elif e == config.n_epoch - 1:
            print("save last model")
            with open(config.train_path, mode='a') as f:
                for i in range(len(train_loss)):
                    f.write('{}, {}\n'.format(i+1, train_loss[i]))
            torch.save(model.state_dict(), config.model_path.replace('epoch', 'last'))
        else:
            if pred > val_loss[-1]:
                print("save best model")
                pred = val_loss[-1]
                with open(config.train_path, mode='a') as f:
                    for i in range(len(train_loss)):
                        f.write('{}, {}\n'.format(i+1, train_loss[i]))
                torch.save(model.state_dict(), config.model_path.replace('epoch', 'best'))


# %% 実行部
if __name__ == '__main__':
    install()
    console = Console()
    try:
        main()
    except:
       console.print_exception()
