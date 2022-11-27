import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 




def plot_fol_distribution(v1, v2, v3):
    """
    For RQ1;
    
    args:
        v1, v2, v3: FOL value vectors
    """
    
    plt.figure(figsize=(8,6))
    plt.title("MNIST / LeNet-5", fontsize=22)
    plt.xlabel("FOL", fontsize=20)
    plt.ylabel("Kernel density(%)", fontsize=20)
    sns.set_context(rc={"lines.linewidth":3})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
#     plt.ylim(0,0.16)
#     plt.yticks([0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16],[0,2,4,6,8,10,12,14,16], fontsize=18)
    
    colormap = ['dodgerblue', 'limegreen', 'r']
    sns.kdeplot(v1, label='Model 1', color=colormap[0])
    fig = sns.kdeplot(v2, label='Model 2', color=colormap[1])
    fig = sns.kdeplot(v3, label='Model 3', color=colormap[2])
    plt.legend(fontsize=20)
    xfig = fig.get_figure()
    xfig.savefig("./mnist-lenet5-fol-distribution.png", dpi=200)
    
    
    
    
def plot_selection(avg1, std1):
    """
    For RQ2; 
    
    args:
        avg1, std1: average and deviation value vectors
    """
    plt.figure(figsize=(8,6))
    plt.title("ATTACK - MNIST / LeNet-5", fontsize=22)
    plt.xlabel("# Percentage of test cases (%)", fontsize=20)
    plt.ylabel("Robustness (%)",fontsize=20)

    xs = [1, 2, 4, 6, 8, 10]
    plt.xticks(xs, xs, fontsize=18)
    plt.yticks(fontsize=18)
    
    colormap = ['r', 'limegreen', 'darkorange', 'dodgerblue']
    
    ## BSET
    avg = np.array(avg1)*100
    std = np.array(std1)*100
    r1 = list(map(lambda x: x[0]-x[1], zip(best_avg, best_std)))
    r2 = list(map(lambda x: x[0]+x[1], zip(best_avg, best_std)))
    plt.plot(xs, best_avg, color=colormap[0], label="BE-ST", marker='o', linewidth=2, markersize=7, alpha=0.8)
    plt.fill_between(xs, r1, r2, color=colormap[0], alpha=0.2)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig("./attack-mnist-lenet5.png", dpi=200)
    
    
    
