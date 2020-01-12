<!--- # Project Title --->
# Deep Learning Based Approach for Compound Type Identification in Sanskrit
Code for the paper titled ["Revisiting the Role of Feature Engineering for Compound Type Identification in Sanskrit"](https://www.aclweb.org/anthology/W19-7503/)

## Requirements:
The following software must be installed on your machine.
* Python 3.5
* Tensorflow 1.13.1
* numpy 
* gensim
* pandas
* scikit-learn

## File organization
* code  : To get results reported in paper, simply run this python file.
* data  : contains data required to run this code
* model : generated model will be stored to this folder

## To run the code:
We have only provided our best word embedding model implementation i.e. FastText.
Go to code/train.py file
```
python train.py
```
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


<!---
<img src="images/vedabaseDF.png"> 

## Categorization of embeddings:

* Word level                            : Word2Vec, glove
* Subword level                         : FastText, Charagram, BPE, LexVec
* Character level                       : charcnn 
* contextual embeddings (Language model): ELMo, BERT, ULMFiT, openAI, FlairEmbeddings

## Evaluation

Parameter details are given in Fasttext ipython notebook in code directory.

To decide max_n parameter, histogram plot of no of characters in word plotted.
So value chosen for experimentation is max_n=10. 

<p align="center">
<img width="400" height="400" src="images/ave_word_plot.png">
</p>

To decide window parameter, histogram plot of, no of words per line plotted.
So window_size chosen is 11. 

<p align="center">
<img width="400" height="400" src="images/word_per_line.png">
  </p>
  
Variation of accuracy as the epoch is changing.

Other parameters were fixed as follows:

Emedding size =100, hidden_layer_sizes=(100),alpha=0.05,learning_rate='adaptive',max_iter=1000,max_n = 10

<p align="center">
<img width="400" height="400" src="images/epoch.png">
    </p>
 
Variation of accuracy as the embedding size is changing.

Other parameters were fixed as follows:

hidden_layer_sizes=(100),alpha=0.05,learning_rate='adaptive',max_iter=1000,max_n = 10 epochs = 5

<p align="center">
<img width="400" height="400" src="images/embedding.png">
</p>
--->

