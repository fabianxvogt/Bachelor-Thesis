import copy
import random
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from utilities.dna_utils import distort_seq, random_window, seq_similarity,base_flip
from utilities.bert_utils import bert_seq

ACTIONS = [0,1,2,3] # No of base flips

# TODO: Clean up

class DNA_Environment(gym.Env):
    environment_name = "DNA Correction Game"

    def __init__(self, dna, environment_dimension=4, deterministic=False):

        self.action_space = spaces.Discrete(environment_dimension)
        self.observation_space = spaces.Box(0, 1, shape=(69,), dtype='float32')
        
        self.dna = dna
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
        self.max_episode_steps = environment_dimension
        self.id = "DNA Correction"
        self.description = "DNA Correction"
        self.environment_dimension = environment_dimension
        self.reward_for_achieving_goal = self.environment_dimension
        self.step_reward_for_not_achieving_goal = -1

        self.deterministic = deterministic

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

        corrected_base = base_flip(error_base, action) #BASES[action] #
        self.corrected_seq += corrected_base

        reward = 0
        if (error_base != actual_base): # Error base
            if (corrected_base == actual_base): # Error corrected
                self.errors_corrected += 1
                reward = 200
            else:
                reward = 10
        else:
            if (actual_base != corrected_base): # New Error made
                self.errors_made += 1
                reward = -30
            else:
                reward = 0

        self.total_reward += reward
        self.index += 1
        next_state = self.bert_states[self.index]

        done = False
        if self.index >= len(self.actual_seq):
            self.total_total_reward += self.total_reward
            self.render()
            done = True


        return np.array(next_state.detach()), reward, done, {}
       
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
