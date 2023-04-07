import numpy as np
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import matplotlib
from matplotlib.ticker import ScalarFormatter

from rich.console import Console
from rich.traceback import install
from rich.progress import track

def graph(label, out_path):
    """
        labelのグラフをプロットする.
    """
    fig, ax = plt.subplots(figsize=(5*1.618,5))
    plt.title("Phoneme Label", fontsize=16, color='black')
    # 軸の設定
    ax.tick_params(labelsize=14, direction='in', length=5, width=2, color='black')
    # x軸の設定
    ax.set_xlabel(r"Time", fontsize=16, color='black')
    ax.xaxis.offsetText.set_fontsize(14)
    # y軸の設定
    ax.set_ylabel(r"Phoneme Classes", fontsize=16, color='black')
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_yaxis().set_major_locator(ptick.MaxNLocator(integer=True))
    # 枠線の設定
    for key in ['top', 'bottom', 'right', 'left']:
        ax.spines[key].set_linewidth(2)
        ax.spines[key].set_color('black')
    # プロット
    im = ax.imshow(label.T, aspect="auto", cmap=plt.get_cmap('binary'))
    # カラーバー
    fig.colorbar(im)
    # 出力
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main(src_dir):
    files = glob.glob(src_dir+'/*.npy', recursive=True)
    for label_file in track(files, description='Plot labels...'):
        label = np.load(label_file)
        out_path = '../out/graph/label/' + os.path.splitext(os.path.basename(label_file))[0] + '.png'
        graph(label, out_path)

if __name__ == '__main__':
    install()
    console = Console()
    try:
       main('../../../01_extract_features/for_pretrain/out/test/label')
    except:
       console.print_exception()
