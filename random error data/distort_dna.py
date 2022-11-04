
from Bio import SeqIO
from Bio.Seq import Seq
from random import random
import math 

def random_error(probability):
    return random() <= probability

def random_base_switch(current_base):
    r = random() * 3
    bases = ['A', 'T', 'G', 'C']
    i = bases.index(current_base)

    new_i = math.floor(i + r + 1)
    new_i = new_i % bases.__len__()
    return bases[new_i]

filename = 'refseq_ds_all_off-frames_fb_DNA_test.fasta'
filepath = 'data/refseq/' + filename

error_probability = 0.01

with open('data/errors/err_dna.fasta', 'w') as f:
    record_no = 1
    for record in SeqIO.parse(filepath, "fasta"):
        id = record.id
        i = 0
        seq_str = str(record.seq)
        if (record_no % 100000 == 0):
            print('distorted ' + str(record_no) + ' records! ...')
        for c in seq_str:
            if (random_error(error_probability)):
                error_base = random_base_switch(c)
                id += ' | ' + str(i) + ': ' + c + ' --> ' + error_base
                seq_str = seq_str[:i] + error_base + seq_str[i+1:] 
            i += 1
        record.id = id
        record.seq = Seq(seq_str)
        SeqIO.write(record, f, "fasta")
        record_no += 1
    f.close()