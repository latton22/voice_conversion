import glob
import torch
import numpy as np

from model import ContextNet
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


def main(input_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    gpu_id = int(sys.argv[2])
    if gpu_id >= 0:
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = False
    device = torch.device("cuda:"+ str(gpu_id) if use_cuda else "cpu")

    model = ContextNet()
    params = torch.load(config.test_model_path)
    model.load_state_dict(params)
    model = model.to(device)

    mspec_list = glob.glob(input_dir+'/*.npy')
    for mspec_file in track(mspec_list, description="Predicting..."):
        src_mspec = np.load(mspec_file)
        model = model.eval()
        with torch.no_grad():
            src_mspec  = torch.from_numpy(src_mspec)
            src_mspec = torch.unsqueeze(src_mspec, dim=0)
            src_mspec  = src_mspec.to(device)
            out_ppg = model(src_mspec)
            out_ppg = torch.squeeze(out_ppg)
            out_ppg = out_ppg.cpu().detach().numpy()
        np.save(out_dir+'/'+os.path.basename(mspec_file), out_ppg)

if __name__ == '__main__':
    install()
    console = Console()
    try:
       main(condif.src_train_dir, '../out/for_post_module/train')
    except:
       console.print_exception()
