import os
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
from sklearn.preprocessing import StandardScaler, PowerTransformer
from datasets import load_from_disk

def noise_reduce_permute(data, args):
    n = len(data['train'])
    if args.data_fraction < 1:
        ids = np.random.choice(n, int(args.data_fraction*n), replace=False)
        data['train'] = data['train'].select(ids)

    if args.noise > 0:
        noisy_ids = np.random.choice(n, int(args.noise*n), replace=False)
        noisy_labels = {idx: l for idx,l in zip(noisy_ids,
            np.random.permutation(data['train'][noisy_ids]['label']))}
        def process(sample, idx):
            if idx in noisy_ids:
                sample['label'] = noisy_labels[idx]
            return sample
        data['train'] = data['train'].map(process, with_indices = True)

    if args.diff_permute:
        diff = np.random.permutation(data['train']['difficulty_class'])
        def process(sample, idx):
            sample['difficulty_class'] = diff[idx]
            return sample
        data['train'] = data['train'].map(process, with_indices = True)
    return data

def filter_columns(data, args):
    cur_cols = data['train'].column_names
    columns = ['label']
    if args.curr == 'dp' and 'diff' in cur_cols:
        columns.append('diff')

    if 'difficulty_class' in cur_cols:
        columns.append('difficulty_class')
    if 'difficulty_score' in cur_cols:
        columns.append('difficulty_score')

    if 'sentence1' in cur_cols:
        text = ['sentence1', 'sentence2']
    elif 't' in cur_cols:
        text = ['t']
    elif 'sentence' in cur_cols and 'addon' in cur_cols:
        text = ['sentence', 'addon']
    elif 'sentence' in cur_cols:
        text = ['sentence']
    columns += text

    if args.eval_class is not None:
        columns.append(args.eval_class)

    for split in data:
        remove_cols = [k for k in data[split].features if not k in columns]
        data[split] = data[split].remove_columns(remove_cols)
    return data, text

def partition(data, diff, args):
    thresholds = [np.percentile(diff, i/args.diff_classes*100) for i in range(args.diff_classes)]
    def assign_class(row):
        diff_class = 0
        val = row['difficulty_score']
        for i in range(args.diff_classes - 1, -1, -1):
            if val >= thresholds[i]:
                diff_class = i
                break
        return {'difficulty_class': diff_class}
    data = data.map(assign_class)
    return data

def process_diff(data, args):
    if args.diff_score is not None:
        data = data.rename_column(args.diff_score, 'difficulty_score')

        scaler = StandardScaler()
        diff = np.array(data['train']['difficulty_score']).reshape(-1,1)
        scaler.fit(diff)
        def scale(row):
            diff = np.array(row['difficulty_score']).reshape(-1,1)
            return {'difficulty_score': scaler.transform(diff).ravel()}
        data = data.map(scale, batched=True)

        diff = data['train']['difficulty_score']
        data = partition(data, diff, args)
    elif args.diff_class is not None:
        data = data.rename_column(args.diff_class, 'difficulty_class')

    return data

def process_labels(sample, n):
    labels = sample['labels']
    np.random.shuffle(labels)
    labels = labels[:n]
    label = np.argmax(np.bincount(labels))
    return {'label': label}


def get_dataloaders(args, tokenizer):
    def tokenize(x):
        if 'sentence1' in x:
            return tokenizer(x['sentence1'], x['sentence2'],
                    truncation=True, max_length = args.max_length, padding='max_length'
                    )
        elif 't' in x:
            return tokenizer(x['t'], 
                    truncation=True, max_length = args.max_length, padding='max_length')
        elif 'sentence' in x:
            return tokenizer(x['sentence'], 
                    truncation=True, max_length = args.max_length, padding='max_length')

    def collate_fn(batch):
        return {k: torch.tensor([x[k] for x in batch]) for k in batch[0].keys()}

    data = load_from_disk(os.path.join(args.data_dir, args.data))

    if args.n_annots > 0:
        labels = data['train']['label']
        data['train'] = data['train'].map(process_labels, fn_kwargs={'n': args.n_annots})
        labels_after = data['train']['label']

    data = process_diff(data, args)

    data = noise_reduce_permute(data, args)

    data, text = filter_columns(data, args)

    data = data.map(tokenize, batched = True, remove_columns = text)

    if args.overfit:
        data['train'] = data['train'].select(range(1))

    sampler = RandomSampler(data['train'])

    train_dataloader = DataLoader(data['train'], args.batch_size,
            sampler=sampler, collate_fn = collate_fn, num_workers=0)
    dev_dataloader = DataLoader(data['dev'], args.batch_size, collate_fn = collate_fn)
    test_dataloader = DataLoader(data['test'], args.batch_size, collate_fn = collate_fn)
    train_dataloader_ns = DataLoader(data['train'], args.batch_size, collate_fn=collate_fn)\
            if args.save_losses else None

    if args.curr == 'dp':
        args.dp_tao = float(np.percentile(data['train']['diff'], 50))

    return train_dataloader, dev_dataloader, test_dataloader,\
            data['dev'], train_dataloader_ns
