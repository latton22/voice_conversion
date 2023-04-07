import glob
import numpy as np

from torch.utils.data import Dataset

import os
import sys
from importlib import import_module
config = sys.argv[1]
config_dir = os.path.dirname(config)
config_bname = os.path.splitext(os.path.basename(config))[0]
sys.path.append(config_dir)
config = import_module(config_bname)

class VCDataset(Dataset):
    def __init__(self, src_dir, tgt_dir, mspec_dim, ppg_dim):
        self.src_files = glob.glob(src_dir+'/*.npy', recursive=True)
        tmp = glob.glob(tgt_dir+'/*.npy', recursive=True)
        self.tgt_files = []
        for src in self.src_files:
            tgt = src.replace('mspec', 'label')
            if tgt in tmp:
                self.tgt_files.append(tgt)
            else:
                print('{} does not exist.'.format(tgt))
        self.mspec_dim = mspec_dim
        self.ppg_dim = ppg_dim
        self.in_data, self.out_data = self.get_datasets()

    def __len__(self):
        return len(self.in_data)

    def __getitem__(self, idx):
        x = self.in_data[idx]
        y = self.out_data[idx]
        return x, y

    def get_datasets(self):
        in_data = [np.load(x) for x in self.src_files]
        out_data = [np.load(x) for x in self.tgt_files]
        for i in range(len(in_data)):
            out_data[i] = np.array([np.where(r==1)[0][0] for r in out_data[i]])
            if in_data[i].shape[1] != out_data[i].shape[0]:
                print('src: {}, tgt: {}'.format(in_data[i].shape[1], out_data[i].shape[0]))
                print('src: {}, tgt: {}'.format(os.path.splitext(os.path.basename(self.src_files[i]))[0], os.path.splitext(os.path.basename(self.tgt_files[i]))[0]))
        return in_data, out_data
