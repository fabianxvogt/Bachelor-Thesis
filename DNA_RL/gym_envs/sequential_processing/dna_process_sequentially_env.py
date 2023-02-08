import numpy as np
from gym import spaces
from gym_envs.dna_env import DNA_Env

class DNA_Process_Sequentially_Env(DNA_Env):
    def __init__(self, error_rate, error_seed=None, use_bert_for_masked_lm = False, kmer_shift=0, seq_len=100):
        super().__init__(error_rate, error_seed=error_seed, use_bert_for_masked_lm = use_bert_for_masked_lm, kmer_shift=kmer_shift, seq_len=seq_len)

    def get_observation_space(self):
        return spaces.Box(-1, 1, shape=(len(self.states),len(self.states[0])), dtype='float32')
    
    def get_observation(self):
        return np.array(self.states.detach())
