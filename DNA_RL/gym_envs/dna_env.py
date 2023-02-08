
import gym
import numpy as np
from data.DNA_sequence_manager import DNA_sequence_manager
from utilities.dna_utils import distort_seq, seq_similarity
from utilities.bert_utils import generate_dnabert_states

class DNA_Env(gym.Env):
    environment_name = "DNA Base Game"

    def prepare_sequence(self):
        dna_sample = self.DNA_seq_manager.get_new_sequence().upper()
        error_seq, error_map = distort_seq(dna_sample, self.error_rate, self.error_seed)
        self.states = generate_dnabert_states(error_seq, self.use_bert_for_masked_lm)

        self.actual_seq_og = dna_sample
        self.error_seq_og = error_seq

        # remove first and last element because of kmer tokenization in DNABERT
        shift = self.kmer_shift
        end = -1+shift
        if end == 0 : end = None
        self.actual_seq = dna_sample[1+shift:end]
        self.error_seq = error_seq[1+shift:end]
        self.error_map = error_map[1+shift:end]

    def get_action_space_size(self):
        return NotImplementedError

    def get_action_space(self):
        raise NotImplementedError()

    def get_observation_space(self):
        raise NotImplementedError()


    def __init__(self, error_rate, error_seed=None, use_bert_for_masked_lm = False, kmer_shift=0, seq_len=100):

        self.DNA_seq_manager = DNA_sequence_manager(seq_len+2) # Add 2 more bases to the sequence lenght because they will be ignored later on (kmer tokenization)
        self.error_rate = error_rate
        self.error_seed = error_seed
        self.sequences_processed = 0
        self.use_bert_for_masked_lm = use_bert_for_masked_lm
        self.kmer_shift = kmer_shift
        self.observation_as_dict = False

        self.prepare_sequence()
        self.current_error_seq = self.error_seq
        self.corrected_seq = self.error_seq

        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.index = 0
        self.total_reward = 0
        self.total_total_reward = 0
        self.max_total_reward = -999999
        self.errors = 0

        self.errors_corrected = 0
        self.errors_found = 0
        self.errors_missed = 0
        self.errors_made = 0
        self.corrects_found = 0

        self.errors_corrected_total = 0
        self.errors_found_total = 0
        self.errors_missed_total = 0
        self.errors_made_total = 0
        self.corrects_found_total = 0

        self.actions_total = 0
        self.errors_total = 0
        self.total_steps = 0

        self.history = []
        self.action_rate_history = []
        self.errors_missed_of_total_errors_history = []
        self.errors_made_of_total_actions_history = []


    def reset(self):
        self.prepare_sequence()
        self.current_error_seq = self.error_seq
        self.corrected_seq = self.error_seq

        self.index = 0
        self.total_reward = 0
        self.errors = 0
        
        # Save total stats
        self.errors_corrected_total += self.errors_corrected
        self.errors_found_total += self.errors_found
        self.errors_missed_total += self.errors_missed
        self.errors_made_total += self.errors_made
        self.corrects_found_total += self.corrects_found

        self.errors_found = 0
        self.errors_corrected = 0
        self.errors_missed = 0
        self.errors_made = 0
        self.corrects_found = 0
        self.predicted_error_map = [0]*len(self.actual_seq)

        self.observation_as_dict = False

        return self.get_observation()

    def get_observation(self):
        raise NotImplementedError()

    def apply_action(self, action):
        raise NotImplementedError()

    def calculate_reward(self, action):
        raise NotImplementedError()

    def is_done(self, action):
        raise NotImplementedError()

    def update(self, action):
        raise NotImplementedError()

    def step(self, action):
        self.apply_action(action)
        reward = self.calculate_reward(action)
        self.update(action)
        self.total_reward += reward
        done = False
        if self.is_done(action):
            done = True
            next_state = None
            self.render()
        else:
            next_state = self.get_observation()

        return np.array(next_state), reward, done, {}

    def get_plot_data(self):
        error_rate = self.errors_total/self.total_steps if self.errors_total > 0 else 0.1
        action_rate = self.actions_total/self.total_steps if self.actions_total > 0 else error_rate
        
        return error_rate, action_rate, self.errors_corrected_total, self.errors_found_total, self.errors_missed_total, self.errors_made_total, self.corrects_found_total

    def set_plot_data(self, errors_corrected_total, errors_found_total, errors_missed_total, errors_made_total, corrects_found_total):
        self.errors_corrected_total = errors_corrected_total
        self.errors_found_total = errors_found_total
        self.errors_missed_total = errors_missed_total
        self.errors_made_total = errors_made_total
        self.corrects_found_total = corrects_found_total

        self.errors_total = self.errors_found_total + self.errors_missed_total
        self.actions_total = self.errors_found_total + self.errors_made_total
        self.total_steps = self.errors_found_total + self.errors_missed_total + self.errors_made_total + self.corrects_found_total

    def render(self, mode='human', close=False):
        print("\nSequence length : " + str(len(self.actual_seq)))
        print("Acutal Sequence:       " + self.actual_seq[0:40])
        print("Error Sequence:        " + self.error_seq[0:40])
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
        print("Error: " + str(error_rate))
        if self.total_steps > 0: print("Actions: " +str(round(self.actions_total/self.total_steps if self.actions_total > 0 else error_rate,4)))
        if self.errors_missed_total > 0: print("Corrected/Missed: " +str(round(self.errors_found_total/self.errors_missed_total,4)))
        if self.errors_made_total > 0: print("Corrected/falsified: " +str(round(self.errors_found_total/self.errors_made_total,4)))
        print("Using BERT for masked LM: " + str(self.use_bert_for_masked_lm))