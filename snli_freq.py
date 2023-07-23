from datasets import load_dataset
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from collections import defaultdict
tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
        }
lemmatizer = WordNetLemmatizer()

data = load_dataset('json', data_files={
    name: f'data/snli_1.0/snli_1.0_{name}.jsonl'
    for name in ['train'] })['train']

freqs = defaultdict(int)

def process(row):
    sent1 = row['sentence1'].lower()
    sent2 = row['sentence2'].lower()

    sent1 = pos_tag(word_tokenize(sent1))
    sent2 = pos_tag(word_tokenize(sent2))

    for word, tag in sent1:
        tag = tag
        lemma = lemmatizer.lemmatize(word, tag_dict[tag[0]] if tag[0] in tag_dict else 'n')
        s = "{} {} {}".format(word, lemma, tag)
        freqs[s] += 1

data.map(process)
with open('snli_freq.txt', 'w') as f:
    for k,v in freqs.items():
        f.write("{} {}\n".format(k,v))
