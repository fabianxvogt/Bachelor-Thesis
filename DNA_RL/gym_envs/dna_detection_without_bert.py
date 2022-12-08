import copy
import math
import random
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from utilities.dna_utils import distort_seq, random_window, seq_similarity, base_flip, base_flip_steps
from utilities.bert_utils import bert_seq
from gym_envs.dna_env import DNA_Env

ACTIONS = [0,1] # 0 = OK, 1 = Error
BASES = ['A', 'C', 'G', 'T']

class DNA_Error_Detection_Env_Without_BERT(DNA_Env):
    environment_name = "DNA Error Detection Game"

    def seq_as_states(self, seq):
        states = []
        for i, c in enumerate(seq):
            state = [0,0,0,0]
            state[BASES.index(c)] = 1
            states.append(state)
        return states


    def prepare_input(self, dna_sample):
        error_seq, error_map = distort_seq(dna_sample, self.error_rate, self.error_seed)
        states = self.seq_as_states(error_seq)

        # remove first and last element because of kmer tokenization in BERT
        # shift = self.kmer_shift
        # end = -1+shift
        # if end == 0 : end = None
        # dna_sample = dna_sample[1+shift:end]
        # error_seq = error_seq[1+shift:end]
        # error_map = error_map[1+shift:end]


        return dna_sample, error_seq, error_map, states


    def __init__(self, full_dna, error_rate, error_seed=None, random_processing=True, use_bert_states=True, kmer_shift=0):
        super().__init__(full_dna, error_rate, error_seed, random_processing, use_bert_states=use_bert_states, kmer_shift=kmer_shift)
        self.actual_seq, self.error_seq, self.error_map, self.states = self.prepare_input(self.get_new_sequence()) 
        self.corrected_seq = self.error_seq
        
        self.action_space = spaces.Box(0, 4, shape=(len(self.states),))
        self.observation_space = spaces.Box(0, 1, shape=(len(self.states), len(self.states[0])), dtype='float32')
        #self.dna = dna
        self.id = "DNA Error Detection"
        self.predicted_error_map = [0]*len(self.actual_seq)
        #self.environment_dimension = environment_dimension
        # self.reward_for_achieving_goal = self.environment_dimension
        # self.step_reward_for_not_achieving_goal = -1

        #self.deterministic = deterministic


    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def reset(self):
        
        self.actual_seq, self.error_seq, self.error_map, self.states = self.prepare_input(self.get_new_sequence()) 
        self.corrected_seq = self.error_seq

        # self.actual_seq, self.error_seq, self.error_map, self.bert_states = get_env_input(self.dna) 
        # if not self.deterministic:
        #     self.desired_goal = self.randomly_pick_state_or_goal()
        #     self.state = self.randomly_pick_state_or_goal()
        # else:
        #     self.desired_goal = [0 for _ in range(self.environment_dimension)]
        #     self.state = [1 for _ in range(self.environment_dimension)]
        # self.state.extend(self.desired_goal)
        # self.achieved_goal = self.state[:self.environment_dimension]
        # self.step_count = 0

        self.index = 0
        self.total_reward = 0
        self.errors = 0
        self.errors_made = 0
        self.errors_corrected = 0
        self.errors_found = 0
        self.wrong_errors_found = 0
        self.predicted_error_map = [0]*len(self.actual_seq)


        return np.array(self.states)
    #     return {"observation": np.array(self.bert_states[0].detach()), "desired_goal": np.array(desired_goal),
    #             "achieved_goal": np.array(achieved_goal)}

    # def randomly_pick_state_or_goal(self):
    #     return [random.randint(0, 1) for _ in range(self.environment_dimension)]

    def highscore(self):
        # if (self.total_reward != 0) & (self.total_reward > self.max_total_reward):
        #     self.max_total_reward = self.total_reward
        #print(seq_similarity(self.error_seq, self.corrected_seq))
        sim = seq_similarity(self.error_seq, self.corrected_seq)
        if sim > self.highest_similarity:
            self.highest_similarity = sim
        # if (seq_similarity(self.error_seq, self.corrected_seq) > ):
            print("\nSaving Model...")
            return True
        return False

    def step(self, actions):

        #new_predicted_error_map = action

        new_corrected_seq = ""
        for i, action in enumerate(actions):
            old_base = self.corrected_seq[i]
            new_corrected_seq += base_flip(old_base, math.floor(action))
        
        reward = 0
        done = False
        new_similarity = seq_similarity(new_corrected_seq, self.actual_seq)
        old_similarity = seq_similarity(self.corrected_seq, self.actual_seq)
        if (new_corrected_seq == self.actual_seq):
            done = True
            next_state = None
            reward = 1000
        elif (new_similarity > old_similarity):
            self.corrected_seq = new_corrected_seq
            reward = 100
            next_state = self.seq_as_states(self.corrected_seq)

        elif self.index > 20:
            done = True
            next_state = None
            reward = -10
        else:
            next_state = self.states
            reward = (new_similarity - old_similarity)*10

        
        self.total_reward += reward

        self.index += 1

        if done: self.render()

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
        print("\nAcutal:        " + self.actual_seq[0:40])
        print("Errors:        " + self.error_seq[0:40])
        print("Corrected:     " + self.corrected_seq[0:40])
        # print("Error Map:     " + str(self.error_map[0:40]))
        # print("Predicted Map: " + str(self.predicted_error_map[0:40]))
        print("Similarity:    " + str(round(seq_similarity(self.actual_seq, self.corrected_seq)*100, 2)))
        print("Score: " + str(self.total_reward))
        # print("Errors found: " + str(self.errors_found))
        # print("Wrong Errors: " + str(self.errors_made))
        #print("Using BERT-Model: " + str(self.use_bert_states))
