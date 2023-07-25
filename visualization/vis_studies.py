import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
import sys, os
import argparse
import math
sns.set()
sns.set_style("white")

parser = argparse.ArgumentParser()
parser.add_argument('study')
parser.add_argument('--vis', default='')
parser.add_argument('--name')
parser.add_argument('--scale', type=float, default=1)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=24)
parser.add_argument('--noleg', action='store_false')
args = parser.parse_args()
args.vis = [int(x) for x in args.vis.split(',') if x]

mean = lambda l: sum(l)/len(l) if len(l) > 0 else 0

# import optuna
# fig = optuna.visualization.plot_contour(studies)
# fig = optuna.visualization.plot_param_importances(studies)
# fig = optuna.visualization.plot_slice(studies)
# fig.show()
# exit()

def load_data(study):
    studies = joblib.load(study)
    trials = studies.trials
    trials = [t for t in trials if t.values]
    trials.sort(key = lambda x: x.values[0], reverse=True)

    df = pd.DataFrame([t.params for t in trials])
    df['dev_acc'] = [t.values[0] for t in trials]
    # df['test_acc'] = [mean(t.user_attrs['test_accs']) for t in trials]
    df['best_step'] = [mean(t.user_attrs['best_steps'])/10 for t in trials]
    df.drop_duplicates(['0-c1', '0-c2', '1-c1', '1-c2', '2-c1', '2-c2'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df, trials

# df = df.groupby(['0-c1', '0-c2','1-c1', '1-c2', '2-c1', '2-c2']).mean()\
#         .sort_values('dev_acc', ascending=False).reset_index()

def measure_dev_test_corr():
    from scipy.stats import pearsonr, spearmanr
    a = [t.values[0] for t in trials]
    b = [t.user_attrs['test_acc'] for t in trials]
    sns.scatterplot(a,b)
    plt.show()

def vis1(start, end, title):
    x = torch.arange(0, args.scale, 0.1)
    k = int(math.sqrt(end - start))+1
    fig, ax = plt.subplots(k,k, constrained_layout=True, figsize = (12.8,9.6), dpi = 300)
    markers = {0: 'o', 1: 'X', 2: 'D'}
    names = {0: 'easy', 1: 'med', 2: 'hard'}
    colors = {0: 'tab:green', 1: 'tab:orange', 2: 'tab:red'}
    # fig.suptitle(title)
    ax = ax.ravel()
    for idx,i in enumerate(range(start, end+1)):
        entry = df.loc[i]
        ent_classes = (entry.size - 3) // 2
        ax[idx].axvline(entry['best_step']*args.scale, color='red', alpha=0.5, linewidth = 1.5)
        for j in range(ent_classes):
            y = torch.sigmoid(entry['%d-c1'%j] * (x - entry['%d-c2'%j]))
            if ent_classes == 3:
                label = '%s (%g, %s)'%(names[j], entry['%d-c1'%j],
                            ("%.2g"%entry['%d-c2'%j]).lstrip('0'))
                marker = markers[j]
            else:
                c2_label = ("%.2g"%entry['%d-c2'%j])
                if len(c2_label) > 1:
                    c2_label = c2_label.lstrip('0')
                label = '%d (%g, %s)'%(j, entry['%d-c1'%j], c2_label)
                marker = 'o'

            ax[idx].plot(x, y,
                    color = colors[j],
                    label = label,
                    marker = marker,
                    alpha=0.7, linewidth = 5)

        ax[idx].set_title('%d (%.2f%%)'%(i,
            entry['dev_acc']*100,
            # entry['test_acc']*100
            ))

        if args.noleg:
            ax[idx].legend(fontsize=8, markerscale = 0.5, handlelength = 0)
    plt.savefig("vis/studies/%s.png"%name, dpi = 300)
    # plt.show()

def vis2():
    fig, ax = plt.subplots(3,2, figsize = (10,10))
    for i in range(3):
        for j in range(2):
            x = [t.params['%d-c%d'%(i,j+1)] for t in trials]
            y = [t.values[0] for t in trials]
            df = pd.DataFrame({'x': x, 'y': y})
            df = df.groupby('x').mean()
            ax[i][j].plot(df)
    fig.text(0.05,0.75, "0", size = 20)
    fig.text(0.05,0.45, "1", size = 20)
    fig.text(0.05,0.15, "2", size = 20)
    fig.text(0.28,0.9, "c1", size = 20)
    fig.text(0.72,0.9, "c2", size = 20)
    plt.show()

def vis3():
    sns.histplot([t.values[0] for t in trials], binwidth=0.001)
    plt.show()

def vis4():
    fig, ax = plt.subplots(3,2)
    for i in range(3):
        for j in range(2):
            vals = [t.params['%d-c%d'%(i,j+1)] for t in trials]
            sns.histplot(vals, ax=ax[i][j], binwidth=0.1 if j == 0 else 0.1)
    plt.show()

def vis5():
    # plt.figure(figsize = (6,3), dpi = 300)
    plt.figure(figsize = (3,3), dpi = 300)
    # plt.figure(figsize = (12,8), dpi = 300)
    plot_df = pd.DataFrame()
    x = torch.arange(0, args.scale, 0.1)

    ent_classes = (df.shape[1] - 2) // 2
    if ent_classes == 3:
        names = {0: 'easy', 1: 'med', 2: 'hard'}
        palette = ['tab:green', 'tab:orange', 'tab:red']
    else:
        names = {i: str(i) for i in range(ent_classes)}
        palette = None
    sizes = [8] * ent_classes
    for idx,i in enumerate(range(0,25)):
        entry = df.loc[i]
        for j in range(ent_classes):
            y = torch.sigmoid(entry['%d-c1'%j] * (x - entry['%d-c2'%j]))
            plot_df = pd.concat((plot_df,
                pd.DataFrame([{'x': float(xx), 'y': float(yy), 'class': names[j]} for xx, yy in zip(x,y)])),
                ignore_index=True)
    sns.lineplot(data=plot_df,
            size = 'class', sizes = sizes, alpha = 0.8,
            x = 'x', y = 'y', hue = 'class', style = 'class', dashes = False, markers = True, palette = palette)
    plt.title(args.name)
    plt.xticks([0,0.5,1],fontsize = 18)
    plt.yticks([0,0.5,1],fontsize = 18)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.xlabel('Time', fontsize = 15)
    plt.ylabel('Weight', fontsize = 15)
    # plt.legend(loc = 4, prop={'size': 17}, framealpha = 0.5)
    plt.gca().get_legend().remove()
    dataset = args.study.split('/')[-1].split('_')[0]
    if dataset == 'snli':
        d = 's'
    elif dataset == 'alcohol':
        d = 't'
    elif dataset == 'cancer':
        d = 'r'
    elif dataset == 'chaosnli':
        d = 'c'
    b = 'b' if 'balanced' in args.study else 'f'
    l = 'l' if 'loss' in args.study else 'e'
    name = d + b + l
    sns.despine()
    plt.savefig('vis/studies/%s.pdf'%name, bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':
    studies = args.study.split(',')
    for study in studies:
        df, trials = load_data(study)
        name = args.name if args.name else study.split('/')[-1]
        if 1 in args.vis:
            title = 'confidence over epochs: toa 25 trials\n(dev_acc), [test_acc], (c1,c2)'
            title = name + '\n' + title
            vis1(args.start, args.end, title)
            # trials = trials[50:]
            # vis1(5, 'conf - epoch plot: (50~74) top trials')
        if 2 in args.vis:
            vis2()
        if 3 in args.vis:
            vis3()
        if 4 in args.vis:
            vis4()
        if 5 in args.vis:
            vis5()

# from c1c2conf import create_json
# cfg = trials[0].params
# cfg = {i: {'c1': cfg['%d-c1'%i], 'c2': cfg['%d-c2'%i]} for i in range(3)}
# create_json(cfg)
