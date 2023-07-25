from datasets import load_from_disk
# import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np
import sys

data = load_from_disk('data/snli_special/train')

fn = sys.argv[1]
loss = np.load(fn)
conf = np.load(fn.replace('loss', 'conf'))

print('--loss')
print(pearsonr(data['entropy'], loss)[0])
print(pearsonr(data['entropy_class'], loss)[0])
if len(conf) > 0:
    print('--conf')
    print(pearsonr(data['entropy'], conf)[0])
    print(pearsonr(data['entropy_class'], conf)[0])
