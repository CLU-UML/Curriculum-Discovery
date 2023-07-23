# HuCurL: Human-Induced Curriculum Discovery

Framwork for curriculum discovery using generalized logistic function (glf) as described in [HuCurl: Human-induced Curriculum Discovery](https://aclanthology.org/2023.acl-long.104/).

## Usage
To train:
`python train.py --data [dataset name] --curr [curriculum name] --diff_score [column name of difficulty score] --diff_score [column name of pre-partitioned difficulty classes / overrides diff_score] --diff_classes [number of difficulty levels]`

To perform curriculum search:
`python curr_search.py --curr glf [args.. (same as train.py)]`

## Datasets
[Cancer Type Dataset](http://clu.cs.uml.edu/data/reddit.tar): This dataset is developed to obtain population-level statistics of cancer patients. It contains 3.8k Reddit posts annotated by at least three annotators for relevance to specific cancer types. If you use this dataset, please cite (Elgaar et al., 2023).

[Alcohol Risk Dataset](http://clu.cs.uml.edu/data/twitter.tar): This dataset is developed to obtain population-level statistics of alcohol use reports through social media. It contains more than 9k tweet, annotated by at least three workers for report of first-person alcohol use, intensity of the drinking (light vs. heavy), context of drinking (social vs. individual), and time of drinking (past, present, or future). If you use this dataset, please cite (Amiri et al., 2018).

## Citation
```
@inproceedings{elgaar-amiri-2023-hucurl,
    title = "{H}u{C}url: Human-induced Curriculum Discovery",
    author = "Elgaar, Mohamed  and
      Amiri, Hadi",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.104",
    pages = "1862--1877",
}
```
