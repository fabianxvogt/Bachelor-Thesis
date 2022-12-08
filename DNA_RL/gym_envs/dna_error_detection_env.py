import copy
import random
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from utilities.dna_utils import distort_seq, random_window, seq_similarity, base_flip, base_flip_steps
from utilities.bert_utils import bert_seq
from gym_envs.dna_env import DNA_Env

ACTIONS = [0,1] # 0 = OK, 1 = Error

class DNA_Error_Detection_Env(DNA_Env):
    environment_name = "DNA Error Detection Game"

    def __init__(self, full_dna, error_rate, error_seed=None, random_processing=True, use_bert_states=True, kmer_shift=0):
        super().__init__(full_dna, error_rate, error_seed, random_processing, use_bert_states=use_bert_states, kmer_shift=kmer_shift)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 1, shape=(len(self.bert_states[0]),), dtype='float32')
        self.actions_total = 0
        self.errors_total = 0
        self.total_steps = 0
        self.total_total_reward = 0
        self.id = "DNA Error Detection"
 
    def step(self, action):
        if action not in ACTIONS:
            return

        if action == 1: 
            self.actions_total+=1
            self.predicted_error_map[self.index] = 1


        self.total_steps += 1

        if self.error_map[self.index] == 1:
            self.errors_total += 1

        reward = self.compute_reward(self.error_map[self.index], action, {})
        self.total_reward += reward

        self.index += 1

        done = False
        if self.index >= len(self.actual_seq):
            self.total_total_reward += self.total_reward
            self.sequences_processed += 1
            done = True
            next_state = None
            self.errors_made_total += self.errors_made
            self.errors_corrected_total += self.errors_found
            self.errors_missed_total += self.errors_missed
            self.corrects_found_total += self.corrects_found
            self.render()
        else:
            next_state = self.bert_states[self.index].detach()
        
        err_rate = 0.0
        if self.errors > 0:
            err_rate = self.errors_found/self.errors
        state = [self.index, self.errors/self.index, err_rate, self.errors_made/self.index]

        return np.array(next_state), reward, done, {}

    def goal_achieved(self, next_state):
        return (self.error_map == self.predicted_error_map).all()#next_state[:self.environment_dimension] == next_state[-self.environment_dimension:]

    def compute_reward(self, is_error, is_error_prediction, info):
        
        base_reward_error_found = 1000
        
        error_rate = self.errors_total/self.total_steps if self.errors_total > 0 else 0.1
        action_rate = self.actions_total/self.total_steps if self.actions_total > 0 else error_rate

        errors_found_missed_ratio = self.errors_corrected_total/(self.errors_corrected_total+self.errors_missed_total) if self.errors_corrected_total > 0 else error_rate
       

        errors_made_DIV_correct_found = self.errors_made_total/self.corrects_found_total if self.corrects_found_total > 0 else 1/action_rate
        errors_found_DIV_errors_missed = self.errors_corrected_total/self.errors_missed_total if self.errors_missed_total > 0 else 1/action_rate
        reward = 0
        if is_error == 1:
            self.errors += 1
            if is_error_prediction == 1:
                reward = base_reward_error_found#90/((self.errors_corrected_total/self.errors_made_total)) if self.errors_made_total > 100 else 800 
                self.errors_found += 1
            else:
                reward = -base_reward_error_found*errors_found_DIV_errors_missed
                self.errors_missed += 1
        else:
            if is_error_prediction == 1:
                reward = -base_reward_error_found*errors_found_DIV_errors_missed*(errors_found_missed_ratio) #(action_rate/error_rate)
                self.errors_made += 1 # #error_found_reward*((self.errors_corrected_total/self.errors_made_total)-0.05) if self.errors_made_total > 100 else 10
            else:
                reward = base_reward_error_found*errors_found_DIV_errors_missed*(errors_found_missed_ratio)*errors_made_DIV_correct_found
                self.corrects_found += 1

        return reward 

    def render(self, mode='human', close=False):
        print("\nAcutal:        " + self.actual_seq[0:40])
        print("Errors:        " + self.error_seq[0:40])
        print("Error Map:     " + str(self.error_map[0:40]))
        print("Predicted Map: " + str(self.predicted_error_map[0:40]))
        print("Similarity:    " + str(round(seq_similarity(self.error_map, self.predicted_error_map)*100, 2)))
        print("Score: " + str(self.total_reward))
        self.total_total_reward += self.total_reward
        print("Total Score: " + str(self.total_total_reward))
        print("No of Errors: " + str(self.error_map.count(1)))
        print("Errors found: " + str(self.errors_found))
        print("Wrong Errors: " + str(self.errors_made))
        error_rate = self.errors_total/self.total_steps if self.errors_total > 0 else 0.1
        error_made_DIV_error_found = self.errors_missed_total/self.errors_corrected_total if self.errors_corrected_total > 0 else 1
        print("Error: " + str(error_made_DIV_error_found*error_rate+error_rate if self.errors_made_total > 0 else 1))
        if self.sequences_processed > 0: print("Actions: " +str(round((self.errors_corrected_total+self.errors_made_total)/(self.sequences_processed*100),4)))
        if self.errors_missed_total > 0: print("Corrected/Missed: " +str(round(self.errors_corrected_total/self.errors_missed_total,4)))
        if self.errors_made_total > 0: print("Corrected/falsified: " +str(round(self.errors_corrected_total/self.errors_made_total,4)))
        print("Using BERT-Model: " + str(self.use_bert_states))
