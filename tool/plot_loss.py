import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import matplotlib
import os
import glob
from matplotlib.ticker import ScalarFormatter
from rich.progress import track
from rich.console import Console
from rich.traceback import install

def graph(train_loss, valid_loss, out_dir):
    """
        lossのグラフをプロットする.
    """
    _, ax = plt.subplots(figsize=(5*1.618,5))
    # 軸の設定
    ax.tick_params(labelsize=14, direction='in', length=5, width=2, color='black')
    # x軸の設定
    ax.set_xlabel("Iteration", fontsize=16, color='black')
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().set_major_locator(ptick.MaxNLocator(integer=True))
    # y軸の設定
    plt.ylim(2, 4)
    ax.set_ylabel("Loss", fontsize=16, color='black')
    ax.yaxis.offsetText.set_fontsize(14)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # 10の累乗表示
    # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))

    # 枠線の設定
    for key in ['top', 'bottom', 'right', 'left']:
        ax.spines[key].set_linewidth(2)
        ax.spines[key].set_color('black')
    nparray = np.array(train_loss.iloc[:,2])
    ax.plot(nparray, marker='o', markersize=3, label="Train-loss", color='orangered', linestyle='None')

    y = valid_loss.iloc[:,2]
    lim = valid_loss.iloc[-2,1]
    x = [(valid_loss.iloc[i,0]-1)*lim + valid_loss.iloc[i,1] for i in range(len(valid_loss))]
    ax.plot(x, y, marker='x', markersize=4, label="Valid-loss", color='deepskyblue')
    # 判例の設定
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black',
          fontsize=14, markerscale=0.8, fancybox=False)
    # 出力
    filename = out_dir + '/loss.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    train_path = glob.glob('../out/train_loss.csv')
    valid_path = glob.glob('../out/valid_loss.csv')
    train_loss = pd.read_csv(train_path[0], sep=', ', engine='python')
    valid_loss = pd.read_csv(valid_path[0], sep=', ', engine='python')
    graph(train_loss, valid_loss, '../out/graph')

# %% 実行部
if __name__ == '__main__':
    # install()
    console = Console()
    try:
       main()
    except:
       console.print_exception()
