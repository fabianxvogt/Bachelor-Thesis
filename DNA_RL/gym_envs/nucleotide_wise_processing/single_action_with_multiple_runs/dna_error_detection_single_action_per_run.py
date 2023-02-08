

from utilities.dna_utils import seq_similarity
from gym_envs.nucleotide_wise_processing.single_action_with_multiple_runs.dna_single_action_per_run_env import DNA_Single_Action_Per_Run_Env
from gym_envs.dna_error_detection_env import DNA_Error_Detection_Env

class DNA_Error_Detection_Single_Action_Per_Run_Env(DNA_Error_Detection_Env, DNA_Single_Action_Per_Run_Env):

    def __init__(self, error_rate, error_seed=None, use_bert_for_masked_lm=False, kmer_shift=0, seq_len=100):
        super().__init__(error_rate, error_seed, use_bert_for_masked_lm=use_bert_for_masked_lm, kmer_shift=kmer_shift, seq_len=seq_len)
        self.error_detection_probabilities = [0]*len(self.states)

    def apply_action(self, action):
        self.error_detection_probabilities[self.index] = action

        if self.index >= len(self.states)-1:
            highest_probability_index = self.error_detection_probabilities.index(
                max(self.error_detection_probabilities)
            )

            self.predicted_error_map[highest_probability_index] = 1

    def calculate_reward(self, action):
        if self.index >= len(self.states)-1:
            detected_error_index = self.predicted_error_map.index(1)
            if self.error_map[detected_error_index] == 1: 
                return 10
            else:
                return -10
        return 0

    def update(self, action):
        if self.index >= len(self.states)-1:
            detected_error_index = self.predicted_error_map.index(1)
            if self.error_map[detected_error_index] == 1: 
                self.errors_found += 1
            else:
                self.errors_made += 1
        self.index += 1
            
