from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import networkx as nx
import spacy
import pandas as pd
import ast
import pprint
import json
import glob

data = [json.load(open(file)) for file in glob.glob('./data/*.json')]


#for now we are going to test bert-base-multilingual-cased, but more models are available for huggingface
layers = [-1]
#model = AutoModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
#tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

nlp_pt = spacy.load('pt_core_news_sm')
nlp_it = spacy.load('it_core_news_sm')


#the following should be changed for a machine with GPU
device = torch.device('cpu')

# the following function can be used to retrieve hidden states from a model
def get_hidden_states(encoded, model, layers):
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    # we can add more layers by changing the variable `layers`
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    return output

def get_words_vector(sent, tokenizer, model, layers):
encoded = tokenizer.encode_plus(sent, return_tensors="pt")
# get all token idxs that belong to the word of interest
#token_ids_word = np.where(np.array(encoded.word_ids()) == idx)

return get_hidden_states(encoded, model, layers)
