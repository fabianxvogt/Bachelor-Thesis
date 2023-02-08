from gym_envs.dna_env import DNA_Env

class DNA_Error_Correction_Env(DNA_Env):
    def __init__(self, error_rate, error_seed=None, use_bert_for_masked_lm=False, kmer_shift=0, seq_len=100):
        super().__init__(error_rate, error_seed, use_bert_for_masked_lm=use_bert_for_masked_lm, kmer_shift=kmer_shift, seq_len=seq_len)

    def get_action_space_size(self):
        return 4 # Choose the correct base out of 4 