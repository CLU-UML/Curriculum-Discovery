import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_from_disk
from collections import Counter

sns.set_theme()


data = load_from_disk('data/snli')

tgt_cls = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
ent_cls = {0: 'easy', 1: 'med', 2: 'hard'}

def target_class():
    d= Counter(data['dev']['label'])
    x = [tgt_cls[c] for c in d.keys()]
    y = list(d.values())
    sns.barplot(x, y)
    plt.show()

def ent_class():
    fig, ax = plt.subplots(1,3)

    for tgt in range(3):
        d = [x for x in data['dev'] if x['label'] == tgt]
        d = Counter([x['entropy_class'] for x in d])
        x = [ent_cls[c] for c in d.keys()]
        y = list(d.values())
        sns.barplot(x,y,ax=ax[tgt])
        ax[tgt].set_title(tgt_cls[tgt])
    plt.show()

def num_annotators():
    fig, ax = plt.subplots(3,1)
    for i,split in enumerate(['train', 'dev', 'test']):
        d = Counter([len(x) for x in data[split]['labels']])
        x = list(d.keys())
        y = list(d.values())

        sns.barplot(x,y,ax=ax[i])
    plt.show()


# target_class()
# ent_class()
# num_annotators()
print(data['train']['sentence1'][:3])
print(data['train']['sentence2'][:3])
print(data['train']['label'][:3])
