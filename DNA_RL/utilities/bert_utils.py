
import sys
import torch
import os

sys.path.append(os.path.join(sys.path[0],'utilities','DNABERT', 'src'))
sys.path.append(os.path.join(sys.path[0],'utilities','DNABERT', 'motif'))

from utilities.DNABERT.src.transformers import BertTokenizer, BertForMaskedLM, BertModel
from utilities.DNABERT.motif.motif_utils import seq2kmer
from utilities.DNABERT.src.transformers import BertConfig

from torch.nn.functional import normalize

path_to_model_dir = sys.path[0] + '/utilities/DNABERT/model/3-new-12w-0'

bert_config = BertConfig()
bert_config.vocab_size = 69
bert_config.is_decoder = False
BERT_MODEL = BertModel.from_pretrained(path_to_model_dir, config=bert_config)
BERT_LANG_MODEL = BertForMaskedLM.from_pretrained(path_to_model_dir, config=bert_config) # model for the pretraining
bert_config.is_decoder = True
#BERT_MODEL_DECODE = BertModel.from_pretrained(path_to_model_dir, config=bert_config)
#BERT_LANG_MODEL_DECODE = BertForMaskedLM.from_pretrained(path_to_model_dir, config=bert_config) # model for the pretraining


# def decode_dnabert_states(dnabert_states, use_bert_for_masked_lm=False):
#     input_tensor = torch.tensor(dnabert_states, dtype=torch.float64)

#     # if use_bert_for_masked_lm:
#     #     output = BERT_LANG_MODEL_DECODE(input_tensor)
#     # else:
#     #     output = BERT_MODEL_DECODE(input_tensor)

#     output = output[0][0] 
#     output = output[1:-2]

#     return output

def generate_dnabert_states(seq, use_bert_for_masked_lm=False, masking_kmer_ids=[], normalize_output=True):
    if not seq[-1] == 'Z': 
            seq += 'Z'
    
    tokenizer = BertTokenizer.from_pretrained(path_to_model_dir)

    kmer = seq2kmer(seq,3)
    tokens = tokenizer.tokenize(kmer)
    for i in masking_kmer_ids:
        tokens[i] = '[MASK]'
    ids = tokenizer.convert_tokens_to_ids(tokens)
    inputs = tokenizer.build_inputs_with_special_tokens(ids)
    input_tensor = torch.tensor([inputs], dtype=torch.long)

    if use_bert_for_masked_lm:
        output = BERT_LANG_MODEL(input_tensor)
    else:
        output = BERT_MODEL(input_tensor)

    output = output[0][0] 

    #seq = decode_dnabert_states(output, use_bert_for_masked_lm)

    output = output[1:-2]

    if normalize_output:
        output = normalize(output, p=2.0, dim=1)
    return output
