from utilities.bert_utils import generate_dnabert_states
from gym_envs.dna_error_detection_env import DNA_Error_Detection_Env
from gym_envs.sequential_processing.dna_process_sequentially_env import DNA_Process_Sequentially_Env

from gym import spaces

class DNA_Error_Detection_Sequential_Env(DNA_Error_Detection_Env, DNA_Process_Sequentially_Env):

    def __init__(self, error_rate, error_seed=None, use_bert_for_masked_lm=False, kmer_shift=0, seq_len=100):
        use_bert_for_masked_lm = True # The big hidden states take way to long to process, so we always use the LM
        super().__init__(error_rate, error_seed, use_bert_for_masked_lm=use_bert_for_masked_lm, kmer_shift=kmer_shift, seq_len=seq_len)

    def get_action_space(self):
        return spaces.Discrete(len(self.states))# Detect one error per iteration
        return spaces.MultiBinary(len(self.states)) # Detect all errors at once

    def apply_action(self, action):
        self.predicted_error_map[action] = 1
        self.actions_total += 1


    def calculate_reward(self, action):
        base_reward = 100
        if self.error_map[action] == 1:
            return base_reward * (1-self.error_rate)
        else:
            return - base_reward * self.error_rate

    def update(self, action):
        if self.error_map[action] == 1:
            self.errors_found += 1
        else:
            self.errors_made += 1
        self.index += 1
        self.mask_current_base(action)

    def is_done(self, action):
        return self.index >= len(self.error_seq) * self.error_rate # Stop iterating after the average number of errors in that sequence
            
    def mask_current_base(self, action):
        kmer_ids_to_mask = [action]
        self.states = generate_dnabert_states(self.error_seq_og, self.use_bert_for_masked_lm, kmer_ids_to_mask)
 