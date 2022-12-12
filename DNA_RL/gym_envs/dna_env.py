import copy
import random
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from data.DNA_sequence_manager import DNA_sequence_manager
from utilities.dna_utils import distort_seq, random_window, seq_similarity, get_env_input
from utilities.bert_utils import bert_seq

ACTIONS = [0,1,2,3] # No of base flips

# TODO: Clean up
class DNA_Env(gym.Env):
    environment_name = "DNA Base Game"

    def prepare_input(self):
        dna_sample = self.get_new_sequence()
        error_seq, error_map = distort_seq(dna_sample, self.error_rate, self.error_seed)
        bert_states = bert_seq(error_seq, self.use_bert_states)

        # remove first and last element because of kmer tokenization in BERT
        shift = self.kmer_shift
        end = -1+shift
        if end == 0 : end = None
        dna_sample = dna_sample[1+shift:end]
        error_seq = error_seq[1+shift:end]
        error_map = error_map[1+shift:end]


        return dna_sample, error_seq, error_map, bert_states


    def __init__(self, full_dna, error_rate, error_seed=None, random_processing=True, training_mode=False, use_bert_states=True, kmer_shift=0, seq_len=100):

        self.DNA_seq_manager = DNA_sequence_manager(seq_len)
        self.full_dna = full_dna
        self.error_rate = error_rate
        self.error_seed = error_seed
        self.random_processing = random_processing
        self.sequences_processed = 0
        self.window_size = seq_len
        self.use_bert_states = use_bert_states
        self.kmer_shift = kmer_shift

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 1, shape=(69,), dtype='float32')

        self.saved_inputs = []
        self.saved_inputs_sampled = 0
        self.actual_seq, self.error_seq, self.error_map, self.bert_states = self.prepare_input()
        self.current_error_seq = self.error_seq

        self.index = 0
        self.corrected_seq = ""
        self.predicted_error_map = [0]*len(self.actual_seq)
        self.total_reward = 0
        self.total_total_reward = 0
        self.max_total_reward = -999999
        self.errors = 0
        self.errors_made = 0
        self.errors_corrected = 0
        self.errors_missed = 0
        self.corrects_found = 0
        self.errors_missed_total = 0.0
        self.corrects_found_total = 0.0
        self.errors_corrected_total = 0.0
        self.errors_made_total = 0.0
        self.errors_found = 0
        self.highest_similarity = 0

        self.id = "DNA Base"

    def get_new_sequence(self):
        # dna = ""
        # if self.random_processing:
        #     dna = random_window(self.full_dna, self.window_size+2)
        # else:
        #     i = self.sequences_processed*self.window_size 
        #     dna = self.full_dna[i:i+self.window_size+2]
        
        # return dna
        return self.DNA_seq_manager.get_new_sequence().upper()

    def reset(self):
        # I tried to save the sampled bert states and reuse them for training,
        # so the BERT-Model is not called in each iteration.
        # The list of saves bert states overloaded the RAM though.
        # TODO: fix or remove
        # self.sequences_processed += 1
        if True: #self.saved_inputs_sampled == len(self.saved_inputs):
            self.actual_seq, self.error_seq, self.error_map, self.bert_states = self.prepare_input()
            self.current_error_seq = self.error_seq
            # self.saved_inputs.append([self.actual_seq, self.error_seq, self.error_map, self.bert_states])
            # self.saved_inputs_sampled = 0
        else:
            [self.actual_seq, self.error_seq, self.error_map, self.bert_states] = random.choice(self.saved_inputs)
            self.saved_inputs_sampled += 1

        self.index = 0
        self.total_reward = 0
        self.corrected_seq = ""
        self.errors = 0
        
        self.errors_made = 0
        self.errors_corrected = 0
        self.errors_found = 0
        self.errors_missed = 0
        self.corrects_found = 0
        self.predicted_error_map = [0]*len(self.actual_seq)

        return np.array(self.bert_states[0].detach())

    def step(self):
        raise NotImplementedError()

