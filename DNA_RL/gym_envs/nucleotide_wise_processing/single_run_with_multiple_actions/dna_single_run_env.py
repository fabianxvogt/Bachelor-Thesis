from ..dna_process_by_nucleotide_env import DNA_Process_By_Nucleotide_Env

import numpy as np
from gym import spaces

class DNA_Single_Run_Env(DNA_Process_By_Nucleotide_Env):
    def __init__(self, error_rate, error_seed=None, use_bert_for_masked_lm = False, kmer_shift=0, seq_len=100):
        super().__init__(error_rate, error_seed=error_seed, use_bert_for_masked_lm = use_bert_for_masked_lm, kmer_shift=kmer_shift, seq_len=seq_len)

    def get_action_space(self):
        # For a single run per Sequence, corrections are performed immediately,
        # instead of comparing probabilities for corrections on each base 
        # after a run has finished (see Dna_Single_Action_Per_Run_Env).
        # Action directly correspont to a base correction at a specific index 
        # and therefore, are discrete.
        return spaces.Discrete(self.get_action_space_size()) 

    def is_done(self, action):
        # When doing a single correction run, we can just end when we reached the end 
        # of the sequence
        return self.index >= len(self.states)