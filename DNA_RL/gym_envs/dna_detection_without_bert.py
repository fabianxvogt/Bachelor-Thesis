from gym_envs.dna_env import DNA_Env

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


    def __init__(self, error_rate, error_seed=None, random_processing=True, use_bert_states=True, kmer_shift=0):
        super().__init__(error_rate, error_seed, random_processing, use_bert_states=use_bert_states, kmer_shift=kmer_shift)
        self.states = self.seq_as_states(self.error_seq)
    
    def get_observation(self):
        return self.states[self.index]

