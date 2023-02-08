

from utilities.dna_utils import seq_similarity
from gym_envs.nucleotide_wise_processing.single_run_with_multiple_actions.dna_single_run_env import DNA_Single_Run_Env
from gym_envs.dna_error_correction_env import DNA_Error_Correction_Env

class DNA_Error_Correction_Single_Run_Env(DNA_Error_Correction_Env, DNA_Single_Run_Env):

    def __init__(self, error_rate, error_seed=None, use_bert_for_masked_lm=False, kmer_shift=0, seq_len=100):
        super().__init__(error_rate, error_seed, use_bert_for_masked_lm=use_bert_for_masked_lm, kmer_shift=kmer_shift, seq_len=seq_len)
        self.bases = ['A', 'T', 'G', 'C']

         # If false, the action represents the index of the base in self.bases.
         # If true, the action is used as a shifting length for the correction (action 0 = no shift)
        self.do_base_shift = False

    def shift_base(self, base, shift):
        index = self.bases.index(base)
        new_index = index + shift 
        new_index = new_index % 4
        return self.bases[new_index]

    def apply_action(self, action):

        if action > 0 and self.do_base_shift:
            corrected_base = self.shift_base(self.error_seq[self.index], action)
            self.corrected_seq = self.corrected_seq[:self.index] + corrected_base + self.corrected_seq[self.index + 1:]
            self.predicted_error_map[self.index] = 1
            self.actions_total += action 
        else:
            predicted_base = self.bases[action]
            self.corrected_seq = self.corrected_seq[:self.index] + predicted_base + self.corrected_seq[self.index + 1:]
            if predicted_base != self.error_seq[self.index]:
                self.predicted_error_map[self.index] = 1
                self.actions_total += action 


    def update(self, action):
        self.total_steps += 1
        if self.error_map[self.index]:
            self.errors += 1

        if self.error_map[self.index] and self.predicted_error_map[self.index]: self.errors_found += 1
        if self.error_map[self.index] and not self.predicted_error_map[self.index]: self.errors_missed += 1
        if not self.error_map[self.index] and self.predicted_error_map[self.index]: self.errors_made += 1
        if not self.error_map[self.index] and not self.predicted_error_map[self.index]: self.corrects_found += 1
        if self.error_map[self.index] and self.actual_seq[self.index] == self.corrected_seq[self.index]: self.errors_corrected += 1
        
        self.index+=1

    def is_done(self, action):
        return self.index >= len(self.states)
    
    def calculate_reward(self, action):
        if self.corrected_seq[self.index] != self.actual_seq[self.index]:
            if self.corrected_seq[self.index] == self.actual_seq[self.index]:
                reward = 2430 # Error found and corrected
            elif self.error_map[self.index]:
                reward = 0 # Error found but not corrected
            else:
                reward = -90 # Error made
        elif self.error_map[self.index]:
            reward = -90 # Error missed
        else:
            reward = 10 # Correct found
        return reward

