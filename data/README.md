<!--- # Project Title --->
# Dataset
Description of data files .
We have used same transliteration scheme as that of [Hellwig's](https://github.com/OliverHellwig/sanskrit/blob/master/papers/2018emnlp/code/data_loader.py)
## Corpora
file name | discription
---|---
 train/test.csv  | This is the dataset for compound type classification task.
compound_dic.pickle  | This file is dictionary mapping of compound classification dataset to get word embedding vectors.
Fast_text_features | This folder contains fasttext embedding of classification dataset.

These features can be downloaded from [here](https://drive.google.com/file/d/1N-xI7UZImp_C8eSktQ94dagsUQpDZdrd/view?usp=sharing)

## Sample data
There are four classes. They are represented by integer mapping: Avyaibhav(0), Bahuvrihi(1), Dvandva(2), Tatpurush(3)

Index | Word1 | Word2 | Class
---|--- |---|---
1 | xqDa | vikramaH | 1
2 | prawi | icCakaH | 0
3 | saMmAna | SuSrURA | 2

### Statistics of Corpora contained in Sanskrit
Corpus | No of Verses | No of words
---|---|---
Vedabase|13013  | 190343
DCS|  127376 | 3797593
wiki|78K lines| 663521
Total|  |4651457






