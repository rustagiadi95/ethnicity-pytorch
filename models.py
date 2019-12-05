import globals
import dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import gensim
import re

model_json = json.load(open('config.json'))["model"]

class Nationality_Model(nn.Module):

    def __init__(self) :
        super(Nationality_Model, self).__init__()
        self.ngrams = len(model_json["model_params"]["ngram_params"])
        self.embedding_layers = nn.ModuleList([])
        self.lstm_layers = nn.ModuleList([])
        self.linear = nn.ModuleList([])
        self.dropout = nn.ModuleList([])
        for index, type_of_layer in enumerate(model_json['layers']) :
            if type_of_layer == "Embedding" :
                self.embedding_layers.append(
                    nn.Embedding(
                        model_json["model_params"]["ngram_params"][index % self.ngrams]["vocab_len"],
                        model_json["model_params"]["ngram_params"][index % self.ngrams]["embed_dim"]
                    )
                )
            elif type_of_layer == "Lstm" :
                self.lstm_layers.append(
                    nn.LSTM(
                        model_json["model_params"]["ngram_params"][index % self.ngrams]["embed_dim"],
                        model_json["model_params"]["lstm_output_dim"], batch_first=True
                    )
                )
            elif type_of_layer == 'Dropout' :
                self.dropout.append(nn.Dropout())
            else : 
                self.linear.append(
                    nn.Linear(
                        model_json["model_params"]["lstm_output_dim"] * self.ngrams,
                        model_json["model_params"]["model_output_dim"]
                    )
                )

    def forward(self, x):
        li_embed = [self.embedding_layers[i](x[i]) for i in range(len(x) - 1)]
        lstm_output = [self.run_cell(li_embed[i], self.lstm_layers[i], self.dropout[i], x[3]) for i in range(len(x) - 1)]
        linear_input = torch.cat(lstm_output, dim = 1)
        out = self.linear[0](linear_input)
        return F.softmax(out, dim = 1)

    def run_cell(self, tensor, layer, dropout, input_len):
        output, _ = layer(tensor)
        spread_len = torch.arange(0, input_len.shape[0]).float().to(globals.device) * model_json["model_params"]['max_time_steps'] + input_len - 1
        gathered_outputs = torch.index_select(output.reshape(-1, model_json["model_params"]["lstm_output_dim"]), 0, spread_len.long())
        return gathered_outputs

def initialize_embeddings(embedding_layers, device):

    idx2gram_mapping = {
        0:'unigram', 1:'bigram', 2:'trigram'
    }

    for ngram, items in enumerate(embedding_layers):
        print('Intializing embeddings for {}'.format(idx2gram_mapping[ngram]))
        file = open(globals.paths["data_cleaned_train"], 'r')
        sentences = []
        for line in file.readlines() :
            name, country = line[:-1].split('\t')
            name = re.sub(r'\ufeff', '', name)
            sentences.append([''.join(list(items)) for items in zip(*[name[i:] for i in range(ngram + 1)])])
        temp = np.zeros((items.num_embeddings, items.embedding_dim))
        # print(sentences)
        model = gensim.models.Word2Vec(sentences, size = items.embedding_dim, window=5, min_count=1, workers=3, iter=5)
        ngram2idx, idx2ngram = dataset.create_ngrams(idx2gram_mapping[ngram])

        for i in range(2, len(idx2ngram)):
                if idx2ngram[i] in model:
                    temp[i] = model[idx2ngram[i]]
        temp[1] = np.ones((items.embedding_dim))
        items.weight.data = torch.Tensor(temp).to(device)