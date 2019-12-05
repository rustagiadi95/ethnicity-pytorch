import  globals
import re
import numpy as np
import json

from torch.utils.data import Dataset, DataLoader

model_json = json.load(open('config.json'))["model"]

def create_ngrams(ngram = 'unigram') :
    path = globals.paths[ngram.replace('gram', '') + '2idx_path']
    f = open(path, 'r')
    ngram2idx = {'<UNK>' : 0, '<PAD>' : 1}
    idx2ngram = {0 : '<UNK>', 1 : '<PAD>'}
    for line in f.readlines():
        ngrams, idx = line[:-1].split('\t')
        ngram2idx[ngrams] = int(idx) + 2
        idx2ngram[int(idx) + 2] = ngrams

    return ngram2idx, idx2ngram

def create_country_dict(entity = 'idx'):
    path = globals.paths['country_to_'+entity]
    f = open(path, 'r')
    country2entity = {}
    entity2country = {}
    if entity == 'idx' :
        for idx, line in enumerate(f.readlines()):
            country, _ = line[:-1].split('\t')
            country2entity[country] = idx
            entity2country[idx] = country

        return country2entity, entity2country

    else :
        for line in f.readlines():
            country, ethnicity = line[:-1].split('\t')
            country2entity[country] = ethnicity
        return country2entity    

def create_dataloader(mode='train', batch_size = 1, shuffle=True):
    dataset = EthnicityDataset(mode, batch_size)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader


unigrams = create_ngrams('unigram')
bigrams = create_ngrams('bigram')
trigrams = create_ngrams('trigram')
countrydict = create_country_dict('idx')

class EthnicityDataset(Dataset) :
    def __init__(self, mode, batch_size) :
        self.batch_size = batch_size
        path = globals.paths['data_cleaned_'+mode]
        data = open(path, 'r')
        self.data = []
        for line in data.readlines() :
            name, country = line[:-1].split('\t')
            name = re.sub(r'\ufeff', '', name)
            self.data.append((name, country))

    def __len__(self) :
        return len(self.data)

    def __getitem__(self, idx) :
        inp = self.data[idx]
        unigram = [unigrams[0][c1[0]] if c1[0] in unigrams[0] else 0 
                   for c1 in zip(*[inp[0][i:] for i in range(1)])]
        bigram = [bigrams[0][c1+c2] if c1+c2 in bigrams[0] else 0
                  for c1, c2 in zip(*[inp[0][i:] for i in range(2)])]
        trigram = [trigrams[0][c1+c2+c3] if c1+c2+c3 in trigrams[0] else 0 
                   for c1, c2, c3 in zip(*[inp[0][i:] for i in range(3)])]
        label = countrydict[0][inp[1]]
        
        
        for i in range(model_json["model_params"]["max_time_steps"] - len(unigram)) :
            unigram.append(1)
        for i in range(model_json["model_params"]["max_time_steps"] - len(bigram)) :
            bigram.append(1)
        for i in range(model_json["model_params"]["max_time_steps"] - len(trigram)) :
            trigram.append(1)
        
        return np.array(unigram), np.array(bigram), np.array(trigram), len(inp[0]), label
        


