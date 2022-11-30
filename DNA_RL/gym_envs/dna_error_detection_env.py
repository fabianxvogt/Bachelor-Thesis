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
        #self.dna = dna
        #self.actual_seq, self.error_seq, self.error_map, self.bert_states = get_env_input(dna) 
        # self.index = 0
        # self.corrected_seq = ""
        #self.predicted_error_map = [0]*len(self.actual_seq)
        # self.total_reward = 0
        # self.total_total_reward = 0
        # self.max_total_reward = -999999
        # self.highest_similarity = 0
        # self.errors = 0
        # self.errors_found = 0
        # # self.wrong_errors_found = 0

        # self.seed()
        # self.reward_threshold = 0.0
        # self.trials = 50
        # self.max_episode_steps = environment_dimension
        self.id = "DNA Error Detection"
        #self.environment_dimension = environment_dimension
        # self.reward_for_achieving_goal = self.environment_dimension
        # self.step_reward_for_not_achieving_goal = -1

        #self.deterministic = deterministic


    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    # def reset(self):

    #     # self.actual_seq, self.error_seq, self.error_map, self.bert_states = get_env_input(self.dna) 
    #     # if not self.deterministic:
    #     #     self.desired_goal = self.randomly_pick_state_or_goal()
    #     #     self.state = self.randomly_pick_state_or_goal()
    #     # else:
    #     #     self.desired_goal = [0 for _ in range(self.environment_dimension)]
    #     #     self.state = [1 for _ in range(self.environment_dimension)]
    #     # self.state.extend(self.desired_goal)
    #     # self.achieved_goal = self.state[:self.environment_dimension]
    #     # self.step_count = 0

    #     self.index = 0
    #     self.total_reward = 0
    #     self.corrected_seq = ""
    #     self.errors = 0
    #     self.errors_made = 0
    #     self.errors_corrected = 0
    #     self.errors_found = 0
    #     self.wrong_errors_found = 0
    #     self.predicted_error_map = [0]*len(self.actual_seq)

    #     state = [0,0,0,0]

    #     return np.array(self.bert_states[0].detach())
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

    def step(self, action):
        if action not in ACTIONS:
            return

        if action == 1: self.predicted_error_map[self.index] = 1

        reward = 0
        if self.error_map[self.index] == 1:
            self.errors += 1
            if action == 1:
                reward = 90
                self.errors_found += 1
            else:
                reward = -90
        else:
            if action == 0:
                reward = 10
            else:
                reward = -10
                self.errors_made += 1
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
        
        err_rate = 0.0
        if self.errors > 0:
            err_rate = self.errors_found/self.errors
        state = [self.index, self.errors/self.index, err_rate, self.errors_made/self.index]

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
        print("Error Map:     " + str(self.error_map[0:40]))
        print("Predicted Map: " + str(self.predicted_error_map[0:40]))
        print("Similarity:    " + str(round(seq_similarity(self.error_map, self.predicted_error_map)*100, 2)))
        print("Score: " + str(self.total_reward))
        print("Errors found: " + str(self.errors_found))
        print("Wrong Errors: " + str(self.errors_made))
        print("Using BERT-Model: " + str(self.use_bert_states))
