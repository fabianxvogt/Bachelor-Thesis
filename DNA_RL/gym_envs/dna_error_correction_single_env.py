import copy
import random
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from utilities.dna_utils import distort_seq, random_window, seq_similarity, base_flip, base_flip_steps
from utilities.bert_utils import bert_seq
from gym_envs.dna_env import DNA_Env

ACTIONS = [0,1,2,3] # 0 = OK, 1 = Error
BASES = ['A', 'C', 'G', 'T']

class DNA_Error_Correction_Single_Env(DNA_Env):
    environment_name = "DNA Error Detection Game"

    def __init__(self, full_dna, error_rate, error_seed=None, random_processing=True, use_bert_states=True, kmer_shift=0):
        super().__init__(full_dna, error_rate, error_seed, random_processing, use_bert_states=use_bert_states, kmer_shift=kmer_shift)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, shape=(len(self.bert_states[0]),), dtype='float32')
        self.id = "DNA Error Detection"
        self.iterations = 0
  
    def highscore(self):
        sim = seq_similarity(self.error_seq, self.corrected_seq)
        if sim > self.highest_similarity:
            self.highest_similarity = sim
        # if (seq_similarity(self.error_seq, self.corrected_seq) > ):
            print("\nSaving Model...")
            return True
        return False

    def step(self, action):
        if action not in ACTIONS:
            return

        #if action != 0: self.predicted_error_map[self.index] = 1

        reward = 0

        actual_base = self.actual_seq[self.index]
        start_error_base = self.error_seq[self.index]
        current_error_base = self.current_error_seq[self.index]
        #corrected_base = BASES[action]
        corrected_base = base_flip(current_error_base, action) 
        self.corrected_seq += corrected_base

        reward = 0
        if (start_error_base == actual_base): # Base had no error in the beginning
            if (current_error_base == actual_base): # Base still has no error
                if (corrected_base == actual_base): # No change was made to the correct base
                    reward = 10
                    self.corrects_found += 1
                else: # A new error got produced in a correct base
                    self.predicted_error_map[self.index] = -1
                    self.errors_made += 1
                    reward = -90
            else: # Base has an error from previous run
                if (corrected_base == actual_base): # Error that was made on previous run was corrected
                    self.predicted_error_map[self.index] = 0
                    reward = 90
                else: # Error from previous run was not corrected
                    reward = -90
        else: # base had error in the beginning
            if (current_error_base == actual_base): # Error was already corrected before
                if (corrected_base == actual_base): # No Error produced
                    reward = 10
                else: # A new error was made in an already corrected base
                    reward = -2430 
                    self.predicted_error_map[self.index] = 0
            else: # Error was not corrected before
                if (corrected_base == actual_base): # Error is now corrected
                    self.predicted_error_map[self.index] = 1
                    self.errors_corrected += 1
                    reward = 2430
                else: # Error was not corrected
                    if (corrected_base == current_error_base): # Error base was not modified
                        reward = -90
                        self.errors_missed += 1
                    else: # Error base was modified but not corrected (Error was found but wrong correction base was chosen)
                        reward = 0

            

        # if (error_base != actual_base): # Error base
        #     if (corrected_base == actual_base): # Error corrected
        #         self.errors_corrected += 1
        #         reward = 2430
        #     else:
        #         if (corrected_base != error_base): #Error found but not corrected
        #             reward = 0
        #         else: # Error not found
        #             reward = -90
        #             self.errors_missed += 1
        # else:
        #     if (actual_base != corrected_base): # New Error made
        #         self.errors_made += 1
        #         reward = -90
        #     else:
        #         reward = 10
        #         self.corrects_found += 1

        if (action != 0):
            self.current_error_seq = self.corrected_seq + self.current_error_seq[self.index+1:]
            self.index = 0
            self.corrected_seq = ""
            self.bert_states = bert_seq(self.current_error_seq)
            self.iterations += 1
        else:
            self.index += 1
        
        done = False
        if self.index >= len(self.actual_seq) or self.iterations > 20:
            #if self.iterations > 20:
                #reward = -10000
            self.iterations = 0
            self.total_total_reward += self.total_reward
            self.sequences_processed += 1
            done = True
            next_state = None
            self.errors_made_total += self.errors_made
            self.errors_corrected_total += self.errors_corrected
            self.errors_missed_total += self.errors_missed
            self.corrects_found_total += self.corrects_found
            self.render()
        else:
            next_state = self.bert_states[self.index].detach()
        
        self.total_reward += reward
        err_rate = 0.0
        if self.errors > 0:
            err_rate = self.errors_found/self.errors
        #state = [self.index, self.errors/self.index, err_rate, self.errors_made/self.index]

        return np.array(next_state), reward, done, {}

    def goal_achieved(self, next_state):
        return (self.error_map == self.predicted_error_map).all()#next_state[:self.environment_dimension] == next_state[-self.environment_dimension:]

    def compute_reward(self, corrected_base, actual_base, info):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        
        if (corrected_base == actual_base).all():
            reward = 10
        else:
            reward = -10
        return reward

    def render(self, mode='human', close=False):
        print("\nAcutal:        " + self.actual_seq[0:40])
        print("Errors:        " + self.error_seq[0:40])
        print("Corrected:     " + self.current_error_seq[0:40])
        print("Error Map:     " + str(self.error_map[0:40]))
        print("Predicted Map: " + str(self.predicted_error_map[0:40]))
        print("Similarity:    " + str(round(seq_similarity(self.error_map, self.predicted_error_map)*100, 2)))
        print("Score: " + str(self.total_reward))
        print("No of Errors: " + str(self.error_map.count(1)))
        print("Errors corrected: " + str(self.errors_corrected))
        print("Wrong Errors: " + str(self.errors_made))
        if self.sequences_processed > 0: print("Actions: " +str(round((self.errors_corrected_total+self.errors_made_total)/(self.sequences_processed*100),4)))
        if self.errors_missed_total > 0: print("Corrected/Missed: " +str(round(self.errors_corrected_total/self.errors_missed_total,4)))
        if self.errors_made_total > 0: print("Corrected/falsified: " +str(round(self.errors_corrected_total/self.errors_made_total,4)))
        print("Using BERT-Model: " + str(self.use_bert_states))
