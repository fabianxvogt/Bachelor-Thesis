from ..dna_process_by_nucleotide_env import DNA_Process_By_Nucleotide_Env

import numpy as np
from gym import spaces

class DNA_Single_Action_Per_Run_Env(DNA_Process_By_Nucleotide_Env):

    def __init__(self, error_rate, error_seed=None, use_bert_for_masked_lm = False, kmer_shift=0, seq_len=100):
        super().__init__(error_rate, error_seed=error_seed, use_bert_for_masked_lm = use_bert_for_masked_lm, kmer_shift=kmer_shift, seq_len=seq_len)

    def get_action_space(self):
        # If we do only one correction per sequence iteration, the action space 
        # has to contain probabilities for each base instead of discrete values. 
        # Only the base with the highest probability will be changed.
        return spaces.Box(0, 1, shape=(1,)) 

    def is_done(self, action):
        """ 
        When doing only a single action per run, it's more difficult 
        to determine the correct 'done'-condition. There's only a limited amount 
        of possibilities for telling the agent when a sequence is finished:
          1) When the base-index reached the end of the sequence
          2) When the agent chooses a 'done'-action (only for discrete action space)
          3) When a fixed amount of sequence-iterations is finished

        In DNA_Single_Action_Per_Run_Env, only a single correction-operation is
        happening on each run. That implies, that multiple iterations over the 
        sequence are necessary in order to correct all the errors. 

        Therefore option 1), determining the 'done'-condition by the base-index,
        does not make much sense.
        Also, option 2) is only possible when the action-space is discrete,
        so that is not an option if the action-space consists of 
        correction-probabilites for each base.

        3) is not great either.. 


        So for now, the only option is to go for the index and let the 
        agent only make a single correction for the most likely error-base
        """
        return self.index >= len(self.states)