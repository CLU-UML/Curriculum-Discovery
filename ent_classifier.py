import sys
import torch
import numpy as np
from torch import nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


data = load_from_disk('data/snli_special_fs1')

def DNN():
    fn = sys.argv[1]
    conf = np.load(fn.replace('loss','conf'))
    conf_dev = np.load(fn.replace('loss','conf_dev'))
    balance = np.load('balance_weight.npz')['ent']

    class EntClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            fc_dim = 512
            self.net = nn.Sequential(
                    nn.Linear(5,fc_dim),
                    nn.ReLU(),
                    nn.Linear(fc_dim,3)
                    )

        def forward(self, x):
            return self.net(x)

    # from train import get_dataloaders
    # train_dataset, dev_dataset = get_dataloaders('data/snli_special_fs1')[-2:]
    train_dataset = data['train']
    dev_dataset = data['dev']
    train_dataset.set_format(type=None,
            columns=['sentence1', 'sentence2', 'label',
                'entropy_class', 'ins_weight', 'feature_set1'])
    dev_dataset.set_format(type=None,
            columns=['sentence1', 'sentence2', 'label',
                'entropy_class', 'ins_weight', 'feature_set1'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cls_weight = torch.tensor(balance).float().to(device)
    model = EntClassifier().to(device)
    crit = nn.CrossEntropyLoss()
    # crit = nn.CrossEntropyLoss(cls_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train_dataset = train_dataset.map(lambda x, i: {'conf': conf[i]}, with_indices=True)
    # dev_dataset = dev_dataset.map(lambda x, i: {'conf': conf[i]}, with_indices=True)

    sample_weights = [balance[y] for y in train_dataset['entropy_class']]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
    train_dataloader = DataLoader(train_dataset, 64, sampler=sampler)
    # train_dataloader = DataLoader(train_dataset, 64)
    dev_dataloader = DataLoader(dev_dataset, 64)

    def evaluate():
        accs = []
        for batch in dev_dataloader:
            # x = batch['conf'].float().unsqueeze(1).to(device)
            x = torch.stack(batch['feature_set1']).float().to(device).T
            y = batch['entropy_class'].to(device)
            out = model(x)
            acc = (y == out.argmax(-1)).float().mean()
            accs.append(acc.item())
        print('dev: %.1f'%(sum(accs)/len(accs)*100))

    for i,batch in enumerate(train_dataloader):
        # x = batch['conf'].float().unsqueeze(1).to(device)
        x = torch.stack(batch['feature_set1']).float().to(device).T
        y = batch['entropy_class'].to(device)
        out = model(x)
        loss = crit(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = (y == out.argmax(-1)).float().mean()
        if i % 10 == 0:
            print("%.2f -- %.1f"%(loss.item(), acc.item()*100))
            evaluate()


    outs = []
    for batch in dev_dataloader:
        # x = batch['conf'].float().unsqueeze(1).to(device)
        x = torch.stack(batch['feature_set1']).float().to(device).T
        y = batch['entropy_class'].to(device)
        out = model(x)
        outs.extend(out.argmax(-1).tolist())
    cm = confusion_matrix(dev_dataset['entropy_class'], outs)
    ConfusionMatrixDisplay(cm).plot()
    plt.title('DNN (feature set 1) with upsample')
    plt.show()

def DT():
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=40, criterion="entropy")

    model.fit(data['train']['feature_set1'], data['train']['entropy_class'])
    outs = model.predict(data['dev']['feature_set1'], data['dev']['entropy_class'])
    cm = confusion_matrix(data['dev']['entropy_class'], outs)
    ConfusionMatrixDisplay(cm).plot()
    plt.title('Decision Tree (feature set 1)')
    plt.show()

DNN()
