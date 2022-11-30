
from Bio import SeqIO
from Bio.Seq import Seq
import random 
import math 
from .bert_utils import bert_seq


BASES = ['A', 'C', 'G', 'T']
ERROR_PROPABILITY = 0.15

def distort_seq(dna, probability, seed = None):
    error_dna = ""
    error_map = [0]*len(dna)
    if seed != None: random.seed(seed)
    for i, c in enumerate(dna):
        if ((c != 'Z') & random_error(probability)):
            error_dna += random_base_switch(c)
            error_map[i] = 1
        else:
            error_dna += c
    return error_dna, error_map

def random_error(probability):
    return random.random() <= probability

def random_base_switch(current_base):
    if current_base == 'Z':
        return current_base
    r = random.random() * 3
    bases = ['A', 'T', 'G', 'C']
    i = bases.index(current_base)

    new_i = math.floor(i + r + 1)
    new_i = new_i % bases.__len__()
    return bases[new_i]

def base_flip(base, steps):
    if steps == 0: return base
    i = BASES.index(base)
    new_i = ( i + steps ) % 4
    return BASES[new_i]

def base_flip_steps(base, new_base):
    i = BASES.index(base)
    i_new = BASES.index(new_base)
    steps = i_new - i
    if steps < 0: steps += 4
    return steps

def base_to_vector(base):
    v = [0,0,0,0]
    i = BASES.index(base)
    v[i] = 1
    return v

def seq_similarity(seq1, seq2):
    if len(seq1) != len(seq2): return 0.0
    base_matches = 0
    for i,c in enumerate(seq1):
        if c == seq2[i]:
            base_matches += 1
    return round(base_matches/len(seq1),2)

def random_window(seqence, length=500):
    i = random.randint(0, len(seqence)-500)
    return seqence[i:i+length]

def predict_error_map(dna):
    pass

def get_env_input(sequence, error_propability = ERROR_PROPABILITY, error_seed = None):
    seq_window = random_window(sequence)
    error_seq, error_map = distort_seq(seq_window, error_propability, error_seed)
    bert_states = bert_seq(error_seq)
    return seq_window, error_seq, error_map, bert_states


