
import sys
import torch
import os

basepath = '/Users/I570101/Documents/Bachelor-Thesis/DNA_RL/utilities/DNABERT'

sys.path.insert(1, basepath+'/src/')
sys.path.insert(1, basepath+'/motif/')
sys.path.insert(1, basepath+'/random_error_data/')

from transformers import BertForMaskedLM,BertTokenizer, BertModel
from motif_utils import seq2kmer

path_to_model_dir = basepath + '/model/3-new-12w-0'
#path_to_model_bin = os.path.join(path_to_model_dir,'pytorch_model.bin')
BERT_MODEL = BertModel.from_pretrained(path_to_model_dir)
BERT_LANG_MODEL = BertForMaskedLM.from_pretrained(path_to_model_dir) # model for the pretraining

#seq = open(basepath+"/rl_data/seq1.txt", "r").read().replace('\n', '').upper()

def bert_seq(seq, use_bert_states = True):
    if not seq[-1] == 'Z': 
            seq += 'Z'
    
    tokenizer = BertTokenizer.from_pretrained(path_to_model_dir)

    kmer = seq2kmer(seq,3)
    tokens = tokenizer.tokenize(kmer)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    inputs = tokenizer.build_inputs_with_special_tokens(ids)
    input_tensor = torch.tensor([inputs], dtype=torch.long)


    if use_bert_states:
        output = BERT_MODEL(input_tensor)
    else:
        output = BERT_LANG_MODEL(input_tensor)

    output = output[0][0] 
    output = output[1:-2]

    return output
