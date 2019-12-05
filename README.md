# ethnicity-pytorch

Implementation of model proposed in 
[Name Nationaltiy Classification with Recurrent Neural Networks (Lee et al., IJCAI 2017)](https://www.ijcai.org/proceedings/2017/0289) 
in PyTorch.

Here is the [Tensorflow implementation](https://github.com/jhyuklee/ethnicity-tensorflow).

### Package Requirements
* python 3.6.4.
* pytorch 1.3.1+cu92.
* re(regex) 2.2.1.
* numpy 1.17.0.
* json 2.0.9.
* tqdm (optional) 4.38.0.
* gensim 3.8.1

### Data
Many thanks to [https://github.com/jhyuklee/ethnicity-tensorflow](https://github.com/jhyuklee/ethnicity-tensorflow) for the data.
A collection of ~10000 sample pairs of names, nationality with ethnicity and ~3000 validation and testing samples of the 
same.

### How to run 
`config.json` can be edited to tweak the model, change the running mode (train/test), change the lr decay rate, etc.
`globals.py` used the change the paths and add / edit global variables.

Note :- In `config.json`, *Vocab_len* keys specify the ngram idx2grams and grams2idx size. *embed_dim* key specify the embedding dimension.

To run the code : python run.py
