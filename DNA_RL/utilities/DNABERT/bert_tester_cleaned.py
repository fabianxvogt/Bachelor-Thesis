#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 20:58:26 2022

@author: ck
"""
import torch
import sys
import os
import time

basepath = '/Users/I570101/Documents/Bachelor-Thesis/DNABERT/'

sys.path.insert(1, basepath+'/src/')
sys.path.insert(1, basepath+'/motif/')

from transformers import BertModel,BertForMaskedLM,BertTokenizer #,BertForSequenceClassification,BertForTokenClassification,BertForPreTraining
from motif_utils import seq2kmer

# path to the predtrained model data
path_to_model_dir = basepath + '/model/3-new-12w-0'

path_to_model_bin = os.path.join(path_to_model_dir,'pytorch_model.bin')

# By loading the model we can see what layers are stored in the file - but we cannot use the model as the architecture 
# (e.g. the forward pass) is not stored here 
#loaded_torch_model = torch.load(path_to_model_bin,map_location=torch.device('cpu'))
#print(loaded_torch_model.keys())

# We can define various models - the most interesting ones are most likely
bert_model = BertModel.from_pretrained(path_to_model_dir) # plain bert model

lm = BertForMaskedLM.from_pretrained(path_to_model_dir) # model for the pretraining


print('Bert Model:')
print(bert_model)
print()
print('-'*30)
print()
print('Language Model')
print(lm)

print()
print('-'*30)
print()
print('Creating input')
print()

# Let's define a testsequence
seq = 'AAAACGTGTATGATTTAGGACCAZ'

# and a tokenizer (converting the k-mer sequence to an input sequence for the model)
tokenizer = BertTokenizer.from_pretrained(path_to_model_dir)



kmer = seq2kmer(seq,3)

print(f'test sequence:\n\t{seq}\n\t lenght: {len(seq)}')
print(f'kmer sequence:\n\t{kmer}\n\twith {len(kmer.split())} k-mers')


# Toekinze the k-mer sequence with tokens in the vocab
tokens = tokenizer.tokenize(kmer)
print(f'token sequence:\n\t{tokens}\n\tlength: {len(tokens)}')
# Convert Tokens to Ids (position in the vocab)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f'id sequence:\n\t{ids}\n\tlength {len(ids)}')
# Add speical tokens
inputs = tokenizer.build_inputs_with_special_tokens(ids)
print(f'final input sequence:\n\t{inputs}')
# Convert to tensor of shape [batch_size, seq_len]
input_tensor = torch.tensor([inputs], dtype=torch.long)
print(f'input tensor:\n\t{input_tensor}')



# alternative method that didn't work out for me - will have a look into it
#input_tensor = torch.tensor(tokenizer.batch_encode_plus([seq], add_special_tokens=True, max_length=512)["input_ids"])


#print(dir(tokenizer))


print()
print('-'*30)
print()
print('Creating outputs')
print()
print('BERT Model')
t1 = time.time()
output = bert_model(input_tensor)
t2 = time.time()

print(f'time for forward pass: {int((t2-t1)*1000)}ms')
print('shape of outputs')
for el in output:
    print(el[0][0])
    print('\t',el.shape)

print()
print('Language Model')
t1 = time.time()
output = lm(input_tensor)
t2 = time.time()
print(f'time for forward pass: {int((t2-t1)*1000)}ms')
print('shape of outputs')
for el in output:
    print(el.shape)
    print(el[0][0])

