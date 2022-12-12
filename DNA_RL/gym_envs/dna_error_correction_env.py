import copy
import random
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym_envs.dna_env import DNA_Env
from utilities.dna_utils import distort_seq, random_window, seq_similarity,base_flip
from utilities.bert_utils import bert_seq

ACTIONS = [0,1,2,3] # No of base flips
BASES = ['A', 'C', 'G', 'T']

# TODO: Clean up

class DNA_Environment(DNA_Env):
    environment_name = "DNA Correction Game"

    def __init__(self, full_dna, error_rate, error_seed=None, random_processing=True, use_bert_states=True, kmer_shift=0, seq_len=100):
        super().__init__(full_dna, error_rate, error_seed, random_processing, use_bert_states=use_bert_states, kmer_shift=kmer_shift, seq_len=seq_len)
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, shape=(768,), dtype='float32')
        
        self.dna = full_dna
        #self.actual_seq, self.error_seq, self.bert_states = get_env_input(dna) 
        self.index = 0
        self.corrected_seq = ""
        self.total_reward = 0
        self.total_total_reward = 0
        self.max_total_reward = -999999
        self.errors_made = 0
        self.errors_corrected = 0
        self.highest_similarity = 0

        self.seed()
        self.reward_threshold = 0.0
        self.trials = 50
        self.max_episode_steps = 4
        self.id = "DNA Correction"
        self.description = "DNA Correction"
        self.environment_dimension = 4
        self.reward_for_achieving_goal = self.environment_dimension
        self.step_reward_for_not_achieving_goal = -1


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def highscore(self):
        sim = seq_similarity(self.error_seq, self.corrected_seq)
        if sim > self.highest_similarity:
            self.highest_similarity = sim
            return True
        return False

    def step(self, action):
        #action = np.argmax(action)
        if action not in ACTIONS:
            return

        #action = 0
        error_base = self.error_seq[self.index]
        actual_base = self.actual_seq[self.index]

        corrected_base = BASES[action]
        #corrected_base = base_flip(error_base, action) #BASES[action] #
        self.corrected_seq += corrected_base

        reward = 0
        if (error_base != actual_base): # Error base
            if (corrected_base == actual_base): # Error corrected
                self.errors_corrected += 1
                reward = 2430
            else:
                if (corrected_base != error_base): #Error found but not corrected
                    reward = 0
                else: # Error not found
                    reward = -90
        else:
            if (actual_base != corrected_base): # New Error made
                self.errors_made += 1
                reward = -90
            else:
                reward = 10

        self.total_reward += reward
        self.index += 1

        done = False
        if self.index >= len(self.actual_seq):
            self.total_total_reward += self.total_reward
            self.render()
            done = True
            next_state = None
        else:
            next_state = self.bert_states[self.index].detach()


        return np.array(next_state), reward, done, {}
       
    def compute_reward(self, corrected_base, actual_base, info):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        
        if (corrected_base == actual_base).all():
            reward = 10
        else:
            reward = -10
        return reward

    def render(self, mode='human', close=False):
        print("\nAcutal:    " + self.actual_seq[0:40])
        print("Errors:    " + self.error_seq[0:40])
        print("Corrected: " + self.corrected_seq[0:40])
        print("Similarity: " + str(round(seq_similarity(self.actual_seq, self.corrected_seq)*100, 2)))
        print("Score: " + str(self.total_reward))
        print("Errors corrected: " + str(self.errors_corrected))
        print("Errors made: " + str(self.errors_made))
