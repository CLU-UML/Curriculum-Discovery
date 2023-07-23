import matplotlib
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import sys
import numpy as np


def name(dataset, curr):
    if dataset[0] == 'a': 
        d = 't'
    elif dataset[0] == 's':
        d = 's'
    elif dataset[:2] == 'ca':
        d = 'r'
    elif dataset[:2] == 'ch':
        d = 'c'
    c = curr[0]
    b = 'D' if 'balanced' in dataset or 'chaos' in dataset else 'F'

    if c == 'n':
        return 'none'
    else:
        return '%s-%s-%s'%(d.upper(),b,c.upper())

cfg_dir = {
        'none': 5,
        'S-D-E': 8,
        'S-D-L': 12,
        'S-F-E': 20,
        'S-F-L': 19,
        'T-D-E': 9,
        'T-D-L': 13,
        'T-F-E': 15,
        'T-F-L': 16,
        'R-D-E': 10,
        'R-D-L': 14,
        'R-F-E': 17,
        'R-F-L': 18,
        'C-D-E': 24,
        'C-D-L': 25,
        }

df = pd.read_csv(sys.argv[1])
df.rename(lambda x: x.split('.')[1] if '.' in x else x,
        axis = 1, inplace = True)
for col in ['data', 'curr', 'ent_cfg']:
    df[col] = df[col].apply(lambda x: x.replace('"', ''))

df = df[df['data'] != '-']
df['ent_cfg'] = df['ent_cfg'].astype('int')
datasets = df['data'].unique()
datasets = [(data, curr) for data in datasets for curr in df['curr'].unique() if curr != 'none']
cfgs = [6] + [cfg_dir[name(d, c)] for d,c in datasets]
df.drop(columns = ['run'], inplace = True)
df.acc = df.acc.astype('float')
df = df.groupby(['data', 'curr', 'ent_cfg']).mean()

full_datasets = [x for x in datasets if not 'balanced' in x[0] or 'chaos' in x[0]]
matrix = [[df.loc[data, curr, cfg].acc if (data, curr, cfg) in df.index else 0 for cfg in cfgs]
        for (data, curr) in datasets]
matrix = np.array(matrix)

none_acc = df.loc[:, :, 5].acc
# nones = np.array([none_acc.loc[data].item() for data, curr in datasets])
# matrix = np.concatenate([nones[:,np.newaxis], matrix], 1)
matrix = matrix / matrix.max(1)[:, np.newaxis] * 100

names = [name(d,c) for d,c in datasets]
col_names = ['(inc.)'] + names

col_sort = matrix.mean(0).argsort()[::-1]
matrix = matrix[:, col_sort]
col_names = [col_names[i] for i in col_sort]

row_sort = matrix.mean(1).argsort()[::-1]
matrix = matrix[row_sort, :]
row_names = [names[i] for i in row_sort]

fig, ax = plt.subplots(2,1, sharex = True,
        figsize = (15,10), dpi = 300,
        gridspec_kw={'height_ratios': [14, 1]})
g = sns.heatmap(matrix,
        ax = ax[0],
        vmin = 95,
        linewidths = 0.1,
        linecolor = 'black',
        annot=False, fmt=".1f",
        xticklabels = col_names,
        yticklabels = row_names, 
        cbar=False,
        square=True
        )
ax[0].set_ylabel("Models")
g.set_yticklabels(g.get_yticklabels(), rotation=45, horizontalalignment='right')

g = sns.heatmap(matrix.mean(0)[np.newaxis,:],
        ax = ax[1],
        annot=True, fmt=".1f",
        vmin = 95,
        vmax = 100,
        linewidths = 0.1,
        linecolor = 'black',
        xticklabels = col_names,
        yticklabels = ['Avg.'],
        cbar=False,
        square=True
        )
ax[1].set_xlabel("Configurations")
g.set_yticklabels(g.get_yticklabels(), rotation=45, horizontalalignment='right')
plt.xticks(rotation = 45)

cax = fig.add_axes([0.765,0.23,0.02,0.64])
cmap = sns.cm.rocket
norm = matplotlib.colors.Normalize(vmin = 95, vmax = 100)
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cax)

plt.savefig('vis/nxn_full.pdf', bbox_inches='tight')

full_ids = [idx for idx in range(matrix.shape[0]) if 'F' in row_names[idx] or 'C' in row_names[idx]]
matrix = matrix[full_ids]

matrix_avg = matrix.mean(0)[np.newaxis,:]

row_names = [name(d,c) for d,c in full_datasets]

sort_ids = matrix_avg.argsort()[0][::-1]
matrix = matrix[:, sort_ids]
matrix_avg = matrix_avg[:, sort_ids]
col_names = [col_names[i] for i in sort_ids]

matrix_avg_rows = matrix.mean(1)
sort_ids_rows = matrix_avg_rows.argsort()[::-1]
matrix = matrix[sort_ids_rows, :]
row_names = [row_names[i] for i in sort_ids_rows]

fig, ax = plt.subplots(2,1, sharex = True,
        # dpi = 300, figsize = (10, 6.2),
        dpi = 300, figsize = (10, 6),
        gridspec_kw={'height_ratios': [8, 1],
            # 'width_ratios': [10, 1]
            })
# ax[1][1].axis('off')

g = sns.heatmap(matrix,
        square = True,
        ax = ax[0],
        vmin = 95,
        linewidths = 0.1,
        linecolor = 'black',
        annot=False, fmt=".1f",
        xticklabels = col_names,
        yticklabels = row_names, 
        cbar=False,
        )
ax[0].set_ylabel("Models")
g.set_yticklabels(g.get_yticklabels(), rotation=45, horizontalalignment='right')

g = sns.heatmap(matrix_avg,
        ax = ax[1],
        annot=True, fmt=".1f",
        linewidths = 0.1,
        linecolor = 'black',
        vmin = 95,
        vmax=100,
        xticklabels = col_names, 
        yticklabels = ['Avg.'],
        cbar=False,
        square=True
        )
ax[1].set_xlabel("Configurations")
g.set_yticklabels(g.get_yticklabels(), rotation=45, horizontalalignment='right')
# g.set_xticklabels(col_names, rotation = 45)
plt.xticks(rotation = 45)

cax = fig.add_axes([0.9,0.26,0.02,0.60])
cmap = sns.cm.rocket
norm = matplotlib.colors.Normalize(vmin = 95, vmax = 100)
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cax)


plt.savefig('vis/nxn.pdf', bbox_inches='tight')
