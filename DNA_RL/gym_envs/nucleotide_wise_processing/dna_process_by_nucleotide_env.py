
import numpy as np
from gym import spaces
from gym_envs.dna_env import DNA_Env

class DNA_Process_By_Nucleotide_Env(DNA_Env):
    def __init__(self, error_rate, error_seed=None, use_bert_for_masked_lm = False, kmer_shift=0, seq_len=100):
        super().__init__(error_rate, error_seed=error_seed, use_bert_for_masked_lm = use_bert_for_masked_lm, kmer_shift=kmer_shift, seq_len=seq_len)

    def get_observation_space(self):
        bert_state_space = spaces.Box(low=0, high=1, shape=(len(self.states[0]),), dtype='float32')
        if self.observation_as_dict:
            return spaces.Dict({
                'bert_state': bert_state_space,
                'sequence_length': spaces.Discrete(len(self.states)+1),
                'current_position': spaces.Discrete(len(self.states)),
                'corrections': spaces.Discrete(len(self.states)+1)
            })
        else:
            return bert_state_space

    def get_observation(self):
        bert_state = np.array(self.states[self.index].detach())
        if self.observation_as_dict:
            return {
                'bert_state': bert_state,
                'sequence_length': len(self.states),
                'current_position': self.index,
                'corrections': self.errors_corrected
            }
        else:
            return bert_state


