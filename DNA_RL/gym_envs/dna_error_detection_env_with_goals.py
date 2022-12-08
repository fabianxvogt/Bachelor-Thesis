
from typing import Optional, OrderedDict, Union
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from utilities.dna_utils import distort_seq, random_window, seq_similarity, base_flip, base_flip_steps
from utilities.bert_utils import bert_seq
from gym_envs.dna_env import DNA_Env
from stable_baselines3.common.type_aliases import GymStepReturn

ACTIONS = [0,1] # 0 = OK, 1 = Error

class DNA_Error_Detection_Env_Goals(gym.Env):
    environment_name = "DNA Error Detection Game"
    def get_new_sequence(self):
        dna = ""
        if self.random_processing:
            dna = random_window(self.full_dna, self.window_size+2)
        else:
            i = self.sequences_processed*self.window_size 
            dna = self.full_dna[i:i+self.window_size+2]
        
        return dna
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

    def __init__(self, full_dna, error_rate, error_seed=None, random_processing=True, use_bert_states=True, kmer_shift=0):
        #super().__init__(full_dna, error_rate, error_seed, random_processing, use_bert_states=use_bert_states, kmer_shift=kmer_shift)
        self.full_dna = full_dna
        self.error_rate = error_rate
        self.error_seed = error_seed
        self.random_processing = random_processing
        self.sequences_processed = 0
        self.window_size = 100
        self.use_bert_states = use_bert_states
        self.kmer_shift = kmer_shift
        
        self.actual_seq, self.error_seq, self.error_map, self.bert_states = self.prepare_input()
        self.predicted_error_map = [0]*len(self.actual_seq)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(0, 1, shape=(len(self.bert_states[0]),), dtype='float32'),
                    "achieved_goal": spaces.Discrete(2),
                    "desired_goal": spaces.Discrete(2),
                }
            )
        self.index = 0
        self.total_reward = 0
        self.corrected_seq = ""
        self.errors = 0
        
        self.errors_made = 0
        self.errors_corrected = 0
        self.errors_found = 0 
        self.total_total_reward = 0
        self.errors_corrected_total = 0.0
        self.errors_made_total = 0.0

        self.id = "DNA Error Detection"
        
        
    def get_obs(self):
        """
        Helper to create the observation.

        :return: The current observation.
        """
        return OrderedDict(
            [
                ("observation", self.bert_states[self.index].detach()),
                ("achieved_goal", self.predicted_error_map[self.index-1]),
                ("desired_goal", self.error_map[self.index-1]),
            ]
        )

    def reset(self):
        self.current_step = 0
        self.actual_seq, self.error_seq, self.error_map, self.bert_states = self.prepare_input()
        self.predicted_error_map = [0]*len(self.actual_seq)
        self.index = 0
        self.total_reward = 0
        self.corrected_seq = ""
        self.errors = 0
        
        self.errors_made = 0
        self.errors_corrected = 0
        self.errors_found = 0 
        return self.get_obs()
    
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
    def compute_reward(
        self, achieved_goal: int, desired_goal: int, _info: str
    ) -> np.float32:
        # As we are using a vectorized version, we need to keep track of the `batch_size`
        if achieved_goal == desired_goal == 1:
            reward = 90
            self.errors_found += 1
        if achieved_goal == desired_goal == 0:
            reward = 10
        if (achieved_goal == 0) & (desired_goal == 1):
            reward = -90
        if (achieved_goal == 1) & (desired_goal == 0):
            reward = -10
            self.errors_made += 1
        # Deceptive reward: it is positive only when the goal is achieved
        # Here we are using a vectorized version
        #distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return float(reward)
    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if action == 1: self.predicted_error_map[self.index] = 1

        obs = self.get_obs()
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], None))
        #done = reward == 0
        self.index += 1
        # Episode terminate when we reached the goal or the max number of steps
        #info = {"is_success": done}
        #done = done or self.current_step >= self.max_steps
        done = False
        if self.index >= len(self.actual_seq):
            self.total_total_reward += self.total_reward
            self.render()
            done = True
            next_state = None
            self.errors_made_total += self.errors_made
            self.errors_corrected_total += self.errors_found
        
        return obs, reward, done, {}



    def goal_achieved(self, next_state):
        return (self.error_map == self.predicted_error_map).all()#next_state[:self.environment_dimension] == next_state[-self.environment_dimension:]

    # def compute_reward(self, corrected_base, actual_base, info):
    #     """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
    #     interface to fit with the open AI gym specifications"""
        
    #     if (corrected_base == actual_base).all():
    #         reward = 10
    #     else:
    #         reward = -10
    #     return reward

    def render(self, mode='human', close=False):
        print("\nAcutal:        " + self.actual_seq[0:40])
        print("Errors:        " + self.error_seq[0:40])
        print("Error Map:     " + str(self.error_map[0:40]))
        print("Predicted Map: " + str(self.predicted_error_map[0:40]))
        print("Similarity:    " + str(round(seq_similarity(self.error_map, self.predicted_error_map)*100, 2)))
        print("Score: " + str(self.total_reward))
        print("Errors found: " + str(self.errors_found))
        print("Wrong Errors: " + str(self.errors_made))
        if self.errors_made_total > 0: print("Running ratio: " +str(round(self.errors_corrected_total/self.errors_made_total,4)))
        print("Using BERT-Model: " + str(self.use_bert_states))
