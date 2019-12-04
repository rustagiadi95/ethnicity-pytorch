import os
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data\\cleaned")
LOG_DIR = os.path.join(BASE_DIR, "logs")
WEIGHTS_DIR = os.path.join(BASE_DIR, "saved_weights")

paths = {
    'uni2idx_path' : os.path.join(DATA_DIR, 'unigram2idx.txt'),
    'bi2idx_path' : os.path.join(DATA_DIR, 'bigram2idx.txt'),
    'tri2idx_path' : os.path.join(DATA_DIR, 'trigram2idx.txt'),
    'data_cleaned_train' : os.path.join(DATA_DIR, "data_cleaned_train"),
    'data_cleaned_test' : os.path.join(DATA_DIR, "data_cleaned_test"),
    'data_cleaned_valid' : os.path.join(DATA_DIR, "data_cleaned_valid"),
    'country_to_idx' : os.path.join(DATA_DIR, "country_to_idx.txt"),
    'country_to_ethnicity' : os.path.join(DATA_DIR, "country_to_ethnicity.txt"),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
