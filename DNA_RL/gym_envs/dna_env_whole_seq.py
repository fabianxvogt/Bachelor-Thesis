import copy
import random
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from utilities.dna_utils import distort_seq, random_window, seq_similarity, get_env_input, base_flip
from utilities.bert_utils import bert_seq

ACTIONS = [0,1,2,3] # No of base flips

class DNA_Env_Whole_Seq(gym.Env):
    environment_name = "DNA Base Game"

    def __init__(self, full_dna, error_rate, error_seed=None, random_processing=True, training_mode=False):

        self.full_dna = full_dna
        self.error_rate = error_rate
        self.error_seed = error_seed
        self.random_processing = random_processing
        self.sequences_processed = 0
        self.window_size = 500

        self.sequence_tries = 0
        self.actual_seq = self.get_new_sequence()
        self.error_seq, self.error_map = distort_seq(self.actual_seq, self.error_rate, self.error_seed)
        self.current_seq = self.error_seq 
        self.bert_states = bert_seq(self.error_seq)

        self.sequence_len = len(self.current_seq)
        self.state_len = len(self.bert_states[0])
        self.actions_per_state = 4

        self.action_space = spaces.Box(0,1, shape=(self.sequence_len, self.actions_per_state))
        self.observation_space = spaces.Box(0, 1, shape=(self.sequence_len, self.state_len,), dtype='float32')

        self.index = 0
        self.corrected_seq = ""
        self.predicted_error_map = [0]*len(self.actual_seq)
        self.total_reward = 0
        self.total_total_reward = 0
        self.max_total_reward = -999999
        self.errors = 0
        self.errors_made = 0
        self.errors_corrected = 0
        self.errors_found = 0
        self.highest_similarity = 0

        self.id = "DNA Base"



    def reset(self):
        self.sequences_processed += 1
        self.actual_seq = self.get_new_sequence()
        self.error_seq, self.error_map = distort_seq(self.actual_seq, self.error_rate, self.error_seed)
        self.current_seq = self.error_seq
        self.bert_states = bert_seq(self.error_seq)
        self.sequence_tries = 0
        self.total_reward = 0
        self.corrected_seq = ""
        self.errors = 0
        self.errors_made = 0
        self.errors_corrected = 0
        self.errors_found = 0
        self.predicted_error_map = [0]*len(self.actual_seq)

        return np.array(self.bert_states.detach())

    def step(self, actions):        
        corrected_seq = ""

        seq_similarity_old = seq_similarity(self.current_seq, self.actual_seq)

        for i, action in enumerate(actions):
            error_base = self.current_seq[i]
            corrected_base = base_flip(error_base, np.argmax(action))
            corrected_seq += corrected_base

        seq_similarity_new = seq_similarity(corrected_seq, self.actual_seq)
        
        done = False
        reward = 0

        if corrected_seq == self.actual_seq: # DONE
            reward = 100
            done = True
        else:
            reward = (seq_similarity_new - seq_similarity_old)*100 - 1

            if seq_similarity_new >= seq_similarity_old: # Do not change state when similarity got lower
                reward += 1
                self.bert_states = bert_seq(corrected_seq)
                self.current_seq = corrected_seq
            


        self.total_reward += reward
        self.sequence_tries += 1

        if self.sequence_tries >= 20: # 20 Tries per sequence
            self.total_total_reward += self.total_reward
            self.render()
            done = True

        return np.array(self.bert_states.detach()), reward, done, {}

    def get_new_sequence(self):
        if self.random_processing:
            return random_window(self.full_dna, self.window_size)
        else:
            i = self.sequences_processed*self.window_size 
            return self.full_dna[i:i+self.window_size]

    def render(self, mode='human', close=False):
        print("\nAcutal:    " + self.actual_seq[0:40])
        print("Errors:    " + self.error_seq[0:40])
        print("Corrected: " + self.current_seq[0:40])
        print("Similarity: " + str(round(seq_similarity(self.actual_seq, self.current_seq)*100, 2)))
        print("Score: " + str(self.total_reward))
        #print("Total: " + str(self.total_total_reward))
        print("Errors corrected: " + str(self.errors_corrected))
        print("Errors made: " + str(self.errors_made))