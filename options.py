import argparse, os

model_names = {
        'base': 'roberta-base',
        'base-bert': 'bert-base-uncased',
        'twitter': 'cardiffnlp/twitter-roberta-base',
        'longformer': 'allenai/longformer-base-4096',
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='.')
    parser.add_argument('--aim_repo', default='.')
    parser.add_argument('--ckpt_dir', default='.')
    parser.add_argument('--preds_dir', default='.')
    parser.add_argument('--data', default='snli_balanced')
    parser.add_argument('--curr', default='none')
    parser.add_argument('--aim_exp', default='default')
    parser.add_argument('--ckpt')
    parser.add_argument('--model_name', default='base')
    parser.add_argument('--num_labels', type=int, default=3)
    parser.add_argument('--glf_cfg')
    parser.add_argument('--anti_curr', action='store_true')
    parser.add_argument('--diff_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n_annots', type=int, default=0)
    parser.add_argument('--val_count', type=int)
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--fbeta', type=float, default=0.5)
    parser.add_argument('--diff_score')
    parser.add_argument('--diff_class')
    parser.add_argument('--add_feats')
    parser.add_argument('--eval_class', default='difficulty_class')
    parser.add_argument('--max_length', type=int, default=160)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--max_grad_norm', type=float, default=0)
    parser.add_argument('--sl_lam', type=float, default=1)
    parser.add_argument('--dp_alpha', type=float, default=0.9)
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--balance_logits', action='store_true')
    parser.add_argument('--diff_permute', action='store_true')
    parser.add_argument('--sel_bp', action='store_true')
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--burn_in', type=float, default=0)
    parser.add_argument('--burn_out', type=float, default=0.1)
    parser.add_argument('--sl_mode', default='avg')
    parser.add_argument('--spl_mode', default='easy')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--save_losses', action='store_true')
    parser.add_argument('--seed', default = '0')
    parser.add_argument('--study_name', default='test_study.pkl')
    parser.add_argument('--study_dir', default='studies')
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--data_fraction', type=float, default=1.0)
    args = parser.parse_args()
    args.seed = [int(x) for x in args.seed.split(',')]
    args.model_name = model_names.get(args.model_name, args.model_name)
    os.makedirs(args.ckpt_dir, exist_ok = True)
    assert args.burn_in + args.burn_out <= 1
    return args
