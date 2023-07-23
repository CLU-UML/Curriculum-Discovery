import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datasets import load_from_disk
from scipy.stats import pearsonr, spearmanr
sns.set_theme()

data = load_from_disk('data/snli/test')

losses = np.load('losses.npy')
entropy = np.array(data['entropy'])

ids = np.argsort(losses)
ids_rev = ids[::-1]

labels = data['labels']
labels = [Counter(l).most_common(1)[0][0] for l in labels]
# labels = data['gold_label']

df = pd.DataFrame({"losses": losses, "entropy": entropy, 'label': labels})
df.sort_values('losses', inplace=True)

def plot1():
    plt.hist(losses, bins=50)
    plt.title("Losses Histogram")
    plt.show()
    plt.title("Entropy Histogram")
    plt.hist(data['entropy'])
    plt.show()


def plot2():
    print(pearsonr(losses, entropy)[0])
    plt.plot(losses[ids], entropy[ids])
    plt.title("Loss-Entropy Plot")
    plt.xlabel('loss')
    plt.ylabel('entropy')
    plt.show()

def plot3():
    with open('spurious_small.csv', 'w') as f:
        f.write("loss,labels,entropy,gold,sentence1,sentence2\n")
        k = 1000
        # for idx in ids_rev[:k]:
        for idx in ids[:k]:
            idx = int(idx)
            loss = losses[idx]
            sample = data[idx]
            f.write('{:.2f},"{}","{:.2f}","{}","{}","{}"\n'.format(
                loss,
                ",".join(map(str,sample['labels'])),
                sample['entropy'],
                sample['gold_label'],
                sample['sentence1'].replace('"', '""'),
                sample['sentence2'].replace('"', '""'),
                ))

cls_map = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
        '-': 3}
tgt_cls = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
ent_cls = {0: 'easy', 1: 'med', 2: 'hard'}

def plot4():
    x = [0]
    count = [[0] for i in range(3)]
    for idx in ids[:1500]:
        c0 = 1 if cls_map[data['gold_label'][idx]] == 0 else 0
        c1 = 1 if cls_map[data['gold_label'][idx]] == 1 else 0
        c2 = 1 if cls_map[data['gold_label'][idx]] == 2 else 0
        count[0].append(count[0][-1] + c0)
        count[1].append(count[1][-1] + c1)
        count[2].append(count[2][-1] + c2)
        x.append(losses[idx])

    x = range(len(count[0]))
    plt.title('Cumulative count w/ ascending loss')
    plt.xlabel('index')
    plt.ylabel('cumulative count')
    plt.plot(x,count[0], label='entailment')
    plt.plot(x,count[1], label='neutral')
    plt.plot(x,count[2], label='contradiction')
    plt.legend()
    plt.show()

def plot5():
    roll = 20
    plt.plot(df['losses'], df['entropy'].rolling(roll).mean())
    corr = pearsonr(losses, entropy)[0]
    plt.text(6,0.1,'ρ = %.2f'%corr)
    
    df_grouped = df.groupby('entropy').mean()
    print(df_grouped)
    print("Correlation after grouping:", pearsonr(df_grouped.index, df_grouped['losses'])[0])
    plt.title(f"Loss-Entropy Plot (window={roll})")
    plt.xlabel('loss')
    plt.ylabel('entropy')
    plt.show()

def plot6():
    df_grouped = df.groupby('label').mean()
    print(df_grouped)
    print('Correlation between class average loss and average entropy:', pearsonr(df_grouped['losses'], df_grouped['entropy'])[0])

def plot7():
    df_grouped = df.groupby('entropy').count()
    print(df_grouped)
    for e in df['entropy'].unique():
        sub_df = df[df['entropy'] == e]
        print('-'*10)
        print('Entropy:', e)
        print(sub_df.groupby('label').count())

def plot8():
    df_grouped = df.groupby('entropy').mean().reset_index().drop([2,5])
    plt.plot(df_grouped['entropy'], df_grouped['losses'])
    plt.title(f"Entropy-Loss Plot")
    plt.xlabel('entropy')
    plt.ylabel('loss')
    plt.show()

def plot9():
    print(pearsonr(losses, entropy)[0])
    print(pearsonr(losses, data['entropy_class'])[0])
    print(spearmanr(losses, data['entropy_class'])[0])

def plot10():
    fig, ax = plt.subplots(1,3)
    for tgt in range(3):
        ids = [i for i,x in enumerate(data) if x['entropy_class'] == tgt]
        l = [losses[i] for i in ids]

        sns.kdeplot(l, ax=ax[tgt])
        ax[tgt].set_xlabel('loss')
        ax[tgt].set_title(ent_cls[tgt])
    plt.show()


# d = [data['gold_label'][idx] for idx in ids[:300]]
# print(Counter(d))

plot5()
