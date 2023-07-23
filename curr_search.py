import os, torch, optuna, joblib
import tempfile
import numpy as np
from torch import nn
from options import parse_args
from data import get_dataloaders, update_dataloader
from train import Trainer, mean
from transformers import AutoTokenizer
from math import ceil
import multiprocessing as mp

def run(args, device, tokenizer, bests, steps, cfg, n):
    if n % 5 == 0:
        gpu = '0'
    elif n % 5 == 1:
        gpu = '1'
    elif n % 5 == 2:
        gpu = '2'
    elif n % 5 == 3:
        gpu = '3'
    elif n % 5 == 4:
        gpu = '4'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    torch.manual_seed(0)
    np.random.seed(0)
    train_dataloader, dev_dataloader, test_dataloader,\
            dev_dataset, train_dataloader_ns = get_dataloaders(args, tokenizer)
    args.epoch_size = ceil(len(train_dataloader.dataset) / args.batch_size)
    for seed in range(2):
        torch.manual_seed(seed)
        np.random.seed(seed)
        args.seed = seed
        name = next(tempfile._get_candidate_names())
        writer = None
        trainer = Trainer(args, writer, device, train_dataloader, name=name, glf_cfg=cfg)

        print(f'[Starting Training] {name} ({gpu})')
        trainer.train(train_dataloader, dev_dataloader,
                dev_dataset, train_dataloader_ns)
        print('[Testing]')
        best_acc, best_step = trainer.load_best()
        print('Acc:', best_acc)
        trainer.cleanup()

        bests.append(best_acc)
        steps.append(best_step/args.epoch_size)

def objective(trial):
    manager = mp.Manager()
    bests = manager.list()
    steps = manager.list()
    cfg = {i: {'c1': trial.suggest_float('%d-c1'%i, -10, 10, step=2),
        'c2': trial.suggest_float('%d-c2'%i, -0.5, 1.5, step=0.25)} for i in range(args.diff_classes)}
    p = mp.Process(target=run, args=(args, device, tokenizer, bests, steps, cfg, trial.number))
    p.start()
    p.join()

    bests = list(bests)
    steps = list(steps)
    trial.set_user_attr('best_steps', steps)
    trial.set_user_attr('accs', bests)
    return mean(bests)

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    mp.set_start_method('spawn')
    
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, return_token_type_ids=False)

    study_path = os.path.join(args.study_dir, args.study_name + '.pkl')
    saver = lambda study, _: joblib.dump(study, study_path)

    if os.path.isfile(study_path):
        print('[Resuming Study]')
        study = joblib.load(study_path)
    else:
        # sampler = optuna.samplers.CmaEsSampler(n_startup_trials = 30)
        study = optuna.create_study(study_name = args.study_name, direction='maximize')

    study.optimize(objective, n_trials=400, callbacks = [saver], n_jobs = 5)
