import os, torch, joblib
from torch import nn
from options import parse_args
from train import Trainer, get_dataloaders, init_model, init_opt, mean
from torch.utils.data import DataLoader

def predict(train_dataloader, pred_dataloader):
    torch.manual_seed(args.seed)
    model, tokenizer, curr, name, step = init_model(args, device, ent_cfg)
    optimizer = init_opt(model, args)
    crit = nn.CrossEntropyLoss(reduction='none')
    writer = None

    epoch_size = len(train_dataloader)
    trainer = Trainer(model, tokenizer, crit, optimizer, curr, args.epochs,
            writer, name, step, epoch_size, args.debug, device, args)

    print('[Starting Training]')
    trainer.train(train_dataloader, dev_dataloader, test_samples)
    print('[Testing]')
    best_acc, best_step = trainer.load_best()
    res = trainer.evaluate(pred_dataloader, return_pred = True)[2]
    test_acc = res[2]
    preds = res[-1]
    print('Acc:', best_acc, test_acc)
    trainer.cleanup()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:%d'%args.gpu if torch.cuda.is_available() else 'cpu')
    
    _, dev_dataloader, _, test_samples,\
            train_dataset, _ = get_dataloaders(args)

    train_set = train_dataset[0:232]
    print(train_dataset[0])
    print(len(train_set))
