from Bio import SeqIO
from re import finditer, compile
import random

 # Rege

class DNA_sequence_manager():
    def __init__(self, seq_len):
        self.index = 0
        self.sequences = None
        self.input_file = "/Users/I570101/Documents/Bachelor-Thesis/DNA_RL/data/GRCh38_latest_genomic.fa"
        self.fasta_sequences = SeqIO.parse(open(self.input_file),'fasta')
        self.regex = compile(r"[^N]+(?=N[^N]*N)")
        self.seq_len = seq_len

    def get_new_sequence(self, random_select = True):
        if (self.sequences == None or (self.index >= len(self.sequences)-1)):
            self.index = 0
            self.sequences = self.prepare_new_fasta_sequence()
        
        if(random_select):
            i = random.randrange(0, len(self.sequences))
            sequence = self.sequences[i]
        else:
            sequence = self.sequences[self.index]

        self.index += 1
        return sequence

    def prepare_new_fasta_sequence(self):
        name, fasta_seq = self.get_next_fasta_sequence()
        seqs = self.split_fasta_sequence(fasta_seq)
        input_sequences = []
        for seq in seqs:
            input_sequences += self.split_sequence_by_length(seq, self.seq_len)

        return input_sequences

    def get_next_fasta_sequence(self):
        fasta = self.fasta_sequences.__next__()
        name, sequence = fasta.id, str(fasta.seq)

        return name, sequence

    def split_fasta_sequence(self, fasta_sequence):
        matches = finditer(self.regex, fasta_sequence)
        sequences = []
        for match in matches:
            sequences.append(fasta_sequence[match.regs[0][0]:match.regs[0][1]])
        return sequences

    def split_sequence_by_length(self, sequence, length):
        sequences = [sequence[i:i+length] for i in range(0, len(sequence), length)]
        return sequences[:-1]

    

