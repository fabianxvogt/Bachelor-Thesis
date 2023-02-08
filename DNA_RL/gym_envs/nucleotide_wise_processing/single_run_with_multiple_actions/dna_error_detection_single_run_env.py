

from utilities.dna_utils import seq_similarity
from gym_envs.nucleotide_wise_processing.single_run_with_multiple_actions.dna_single_run_env import DNA_Single_Run_Env
from gym_envs.dna_error_detection_env import DNA_Error_Detection_Env

class DNA_Error_Detection_Single_Run_Env(DNA_Error_Detection_Env, DNA_Single_Run_Env):

    def __init__(self, error_rate, error_seed=None, use_bert_for_masked_lm=False, kmer_shift=0, seq_len=100):
        super().__init__(error_rate, error_seed, use_bert_for_masked_lm=use_bert_for_masked_lm, kmer_shift=kmer_shift, seq_len=seq_len)
    
    def apply_action(self, action):
        self.predicted_error_map[self.index] = action
        self.actions_total += action 

    def update(self, action):
        self.total_steps += 1
        if self.error_map[self.index]:
            self.errors += 1

        if self.error_map[self.index] and self.predicted_error_map[self.index]: self.errors_found += 1
        if self.error_map[self.index] and not self.predicted_error_map[self.index]: self.errors_missed += 1
        if not self.error_map[self.index] and self.predicted_error_map[self.index]: self.errors_made += 1
        if not self.error_map[self.index] and not self.predicted_error_map[self.index]: self.corrects_found += 1
        
        self.index+=1

    def is_done(self, action):
        return self.index >= len(self.states)
    
    def calculate_reward(self, action):
        base_reward_error_found = 1000
        
        error_rate = self.errors_total/self.total_steps if self.errors_total > 100 else 0.1
        action_rate = self.actions_total/self.total_steps if self.actions_total > 100 else error_rate

        good_action_ratio = self.errors_corrected_total/(self.errors_made_total+self.errors_corrected_total) if self.corrects_found_total > 0 else action_rate
        
        desired_action_rate = 0.1 + (1-good_action_ratio) / 2

        
        errors_made_DIV_correct_found = self.errors_made_total/self.corrects_found_total if self.corrects_found_total > 0 else error_rate
        errors_found_DIV_errors_missed = self.errors_found_total/self.errors_missed_total if self.errors_missed_total > 0 else error_rate
        errors_found_DIV_errors_made = self.errors_found_total/self.errors_made_total if self.errors_missed_total > 0 else error_rate
        
        reward = 0
        if self.error_map[self.index] == 1:
            enforce_action_multiplier = desired_action_rate/action_rate

            if self.predicted_error_map[self.index] == 1:
                reward = base_reward_error_found*enforce_action_multiplier#90/((self.errors_corrected_total/self.errors_made_total)) if self.errors_made_total > 100 else 800 
            else:
                reward = -base_reward_error_found*errors_found_DIV_errors_missed*enforce_action_multiplier
        else:
            if self.predicted_error_map[self.index] == 1:
                reward = -base_reward_error_found*error_rate/(1-error_rate) #(action_rate/error_rate)# #error_found_reward*((self.errors_corrected_total/self.errors_made_total)-0.05) if self.errors_made_total > 100 else 10
            else:
                reward = base_reward_error_found*(error_rate/(1-error_rate))*errors_made_DIV_correct_found
        return reward 


