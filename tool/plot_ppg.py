import numpy as np
import glob
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import matplotlib
from matplotlib.ticker import ScalarFormatter

from rich.console import Console
from rich.traceback import install
from rich.progress import track

phoneme_class_dir = '../../../01_extract_features/for_pretrain/out/train/phoneme_class.csv'

def graph(logppg, out_path):
    """
        PPGのグラフをプロットする.
    """
    # one-hotにしてみる
    # ppg = np.identity(35)[np.argmax(logppg, axis=1)]
    ppg = np.exp(logppg)

    fig, ax = plt.subplots(figsize=(7*1.618,7))
    # plt.title("PPG", fontsize=16, color='black')
    # 軸の設定
    ax.tick_params(labelsize=14, direction='in', length=5, width=2, color='black')
    # x軸の設定
    ax.set_xlabel(r"Time [s]", fontsize=16, color='black')
    ax.xaxis.offsetText.set_fontsize(14)
    max = len(ppg.T[0])
    max -= max % 200
    x = [max // 4 * i for i in range(5)]
    ticks = [i * 5 / 1000 for i in x]
    plt.xticks(x, ticks)
    # y軸の設定
    phoneme_class = []
    with open(phoneme_class_dir) as f:
        reader = csv.reader(f)
        for r in reader:
            phoneme_class.append(r[1])
    plt.yticks(range(len(phoneme_class)), phoneme_class)
    ax.set_ylabel(r"Phoneme Classes", fontsize=16, color='black')
    # 枠線の設定
    for key in ['top', 'bottom', 'right', 'left']:
        ax.spines[key].set_linewidth(2)
        ax.spines[key].set_color('black')
    # プロット
    im = ax.imshow(ppg.T, aspect="auto", cmap=plt.get_cmap('viridis'), vmin=0, vmax=1, interpolation='none')
    # カラーバー
    fig.colorbar(im)
    # 出力
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main(src_dir):
    out_dir = '../out/graph/ppg/'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    files = glob.glob(src_dir+'/*.npy', recursive=True)[:10]
    for ppg_file in track(files, description='Plot PPGs...'):
        ppg = np.load(ppg_file)
        out_path = out_dir + os.path.splitext(os.path.basename(ppg_file))[0] + '.png'
        graph(ppg, out_path)

if __name__ == '__main__':
    install()
    console = Console()
    try:
       main('../out/ppg_best_model')
    except:
       console.print_exception()
