import json, os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
# sns.set()

def create_json(cfg):
    cfg_num = len([x for x in os.listdir('cfg') if 'c1c2' in x and x.endswith('.json')])
    out_file = 'cfg/c1c2_%d.json'%cfg_num
    with open(out_file, 'w') as f:
        json.dump(cfg, f)
    print("Written to:", out_file)
    return out_file

def vis_json(cfg_num, epochs):
    # plt.figure(figsize = (4, 3.3), dpi = 300)
    plt.figure(figsize = (3.45,2.45), dpi = 300)
    in_file = 'cfg/c1c2_%d.json'%int(cfg_num)
    with open(in_file) as f:
        cfg = json.load(f)
    x = torch.arange(start=0, end=1, step = 0.01)
    x2 = torch.arange(start=0, end=1, step = 0.1)
    for j in cfg.keys():
        c1 = cfg[j]['c1']
        c2 = cfg[j]['c2']
        j = int(j)
        y = torch.sigmoid(c1 * (x - c2))
        y2 = torch.sigmoid(c1 * (x2 - c2))
        plt.plot(x, y,
                alpha=0.4, linewidth = 5)
        plt.plot(x2, y2, marker = markers[j],
                color = colors[j],
                label = '%s (%g, %s)'%(names[j], c1,
                    ("%.2g"%c2).lstrip('0') if c2 != 0 else 0),
                alpha=0.7, linewidth = 0)
    # sns.despine()
    plt.title(None)
    plt.xlabel('Training Progress', fontsize = 10)
    plt.ylabel('Confidence', fontsize = 10)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend(
            # handletextpad = -0.5,
            handlelength = 0.5,
            loc = 4
            )
    # plt.legend()
    # plt.legend(loc = 4, prop={'size': 12})
    plt.savefig('vis/cfg/%s.png'%cfg_num, bbox_inches='tight')
    # plt.show()

def single_plots(cfg_num, epochs):
    in_file = 'cfg/c1c2_%d.json'%int(cfg_num)
    with open(in_file) as f:
        cfg = json.load(f)
    x = torch.arange(start=0, end=1, step = 0.01)
    x2 = torch.arange(start=0, end=1, step = 0.1)
    for j in cfg.keys():
        plt.figure(dpi = 300)
        c1 = cfg[j]['c1']
        c2 = cfg[j]['c2']
        j = int(j)
        y = torch.sigmoid(c1 * (x - c2))
        y2 = torch.sigmoid(c1 * (x2 - c2))
        plt.plot(x, y, 
                alpha=0.9, linewidth = 15, color = colors[j])
        # plt.plot(x2, y2, marker = markers[j],
        #         color = colors[j],
        #         label = '%s (%.1f, %.1f)'%(names[j], c1, c2),
        #         alpha=0.7, linewidth = 0)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    plt.title(None)
    plt.xlabel('Training Progress', fontsize = 13)
    plt.ylabel('Confidence', fontsize = 13)
    plt.legend(prop={'size': 10})
    plt.savefig('ent_cfg.png', bbox_inches='tight')
    # plt.show()

def base_sigmoid(cfg_num, epochs):
    plt.figure(figsize = (8,5.2), dpi = 300)
    in_file = 'cfg/c1c2_%d.json'%int(cfg_num)
    with open(in_file) as f:
        cfg = json.load(f)
    x = torch.arange(start=0, end=1, step = 0.01)
    x2 = torch.arange(start=0, end=1, step = 0.1)
    for j in cfg.keys():
        c1 = cfg[j]['c1']
        c2 = cfg[j]['c2']
        j = int(j)
        y = torch.sigmoid(c1 * (x - c2))
        y2 = torch.sigmoid(c1 * (x2 - c2))
        plt.plot(x, y, color = colors[j],
                alpha=0.5, linewidth = 6)
        plt.plot(x2, y2, marker = markers[j],
                color = colors[j],
                # label = '%s (%.1f, %.1f)'%(names[j], c1, c2),
                label = '(%g, %s)'%(c1, ("%.2g"%c2).lstrip('0')),
                alpha=0.9, linewidth = 0)
    plt.title(None)
    plt.xlabel('Training Progress', fontsize = 20)
    plt.ylabel('Confidence', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(prop={'size': 15},
            handletextpad = -0.5
            )
    # plt.legend()
    plt.savefig('base_sigmoid.png', bbox_inches='tight')
    # plt.show()



if __name__ == '__main__':
    cfg = {
            0: {'c1': 4.0, 'c2': 0.0},
            1: {'c1': -2.0, 'c2': -0.25},
            2: {'c1': 6.0, 'c2': 1.25}
            }
    create_json(cfg)

    # colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green'}
    # names = {0: 'easy', 1: 'med', 2: 'hard'}
    # markers = {0: 'o', 1: 'X', 2: 'D'}
    # vis_json(sys.argv[1],
    #         epochs = int(sys.argv[2]) if len(sys.argv) == 3 else 10)
    # single_plots(sys.argv[1],
    #         epochs = int(sys.argv[2]) if len(sys.argv) == 3 else 10)

    # colors = {0: 'tab:orange', 1: 'tab:orange', 2: 'tab:orange',
    #         3: 'tab:blue', 4: 'tab:blue', 5: 'tab:blue'}
    # names = {0: 'easy', 1: 'med', 2: 'hard'}
    # markers = {0: 's', 1: 'o', 2: 'D', 3: 'v', 4: 'X', 5: '^'}
    # base_sigmoid(sys.argv[1],
    #         epochs = int(sys.argv[2]) if len(sys.argv) == 3 else 10)
