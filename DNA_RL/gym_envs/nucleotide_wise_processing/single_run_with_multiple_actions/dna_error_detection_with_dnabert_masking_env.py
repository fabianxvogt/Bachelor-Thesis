import copy
import math
import random
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from utilities.dna_utils import distort_seq, random_window, seq_similarity, base_flip, base_flip_steps
from utilities.bert_utils import generate_dnabert_states
from gym_envs.dna_env import DNA_Env

from gym_envs.nucleotide_wise_processing.single_run_with_multiple_actions.dna_error_detection_single_run_env import DNA_Error_Detection_Single_Run_Env

ACTIONS = [0,1] # 0 = OK, 1 = Error

STATES = {
    'A': [1,0,0,0],
    'T': [0,1,0,0],
    'G': [0,0,1,0],
    'C': [0,0,0,1]
}

class DNA_Error_Detection_With_Masking_Correction_Env(DNA_Error_Detection_Single_Run_Env):

    def __init__(self, error_rate, error_seed=None, use_bert_for_masked_lm=False, kmer_shift=0, seq_len=20):
        super().__init__(error_rate, error_seed, use_bert_for_masked_lm=use_bert_for_masked_lm, kmer_shift=kmer_shift, seq_len=seq_len)
        self.masked_bert_state = None # Cache last DNABERT-state, that was masked
        self.observation_as_dict = True
        self.actions_on_current_base = 0

    def apply_action(self, action):
        if action:
            if self.predicted_error_map[self.index]: # Was this base already masked? If so, revert it
                self.states[self.index] = self.masked_bert_state # Revert the state change
            else:
                self.mask_current_base()

            self.actions_on_current_base += 1
            self.predicted_error_map[self.index] = 1 - self.predicted_error_map[self.index]
        else:
            self.actions_on_current_base = 0
            if self.predicted_error_map[self.index]:
                self.actions_total += 1 

    def update(self, action):
        # Only update the stats when the agent continues with the next base

        # We don't wanna raise the current base-index if an action was performed,
        # so the agent has a second chance of evaluating the new DNABERT-state
        # that was generated for the masked base (or the reverted DNABERT-state)
        if not action: super().update(action) 

    def calculate_reward(self, action):
        # Only calculate the reward once we proceed to the next base.
        # For each modification step, the reward just is = -1
        if action:
            reward = -2**(self.actions_on_current_base - 1)
            return reward
        else:
            return super().calculate_reward(action)

    def mask_current_base(self):
        self.masked_bert_state = self.states[self.index]
        kmer_ids_to_mask =[self.index+i for i in range(min(len(self.states)-self.index,3))] #[self.index, self.index + 1, self.index + 2]
        self.states = generate_dnabert_states(self.error_seq_og, self.use_bert_for_masked_lm, kmer_ids_to_mask)
 