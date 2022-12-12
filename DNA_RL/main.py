
import sys

import gym
from utilities.dna_utils import distort_seq
from utilities.bert_utils import bert_seq
from gym_envs.dna_error_correction_env import DNA_Environment
from gym_envs.dna_error_detection_env import DNA_Error_Detection_Env
from gym_envs.dna_env_whole_seq import DNA_Env_Whole_Seq
from gym_envs.dna_detection_without_bert import DNA_Error_Detection_Env_Without_BERT
from gym_envs.dna_error_correction_single_env import DNA_Error_Correction_Single_Env
from gym_envs.dna_error_detection_single2_env import DNA_Error_Detection_Single2_Env
from gym_envs.dna_error_detection_env_with_goals import DNA_Error_Detection_Env_Goals
import random

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, DQN
from sb3_contrib import RecurrentPPO

LEARNING_RATE = 0.001

BERT_MODEL = "BERT"
ENV = None
ERROR_RATE = 0.1
SAMPLE_SIZE = 100
MODEL = RecurrentPPO
LSTM_MULTI_POLICY = "MultiInputLstmPolicy"
LSTM_POLICY = "MlpLstmPolicy"
SB3_POLICY = LSTM_POLICY

if MODEL == A2C:
    SB3_POLICY = "MlpPolicy"

MODE = 2
if MODE == 1: # detect
    ENV = DNA_Error_Detection_Env
    MODEL_NAME = "DNA_detection_dyn"
elif MODE == 2: # correct
    ENV = DNA_Environment
    MODEL_NAME = "DNA_correction"
elif MODE == 3:
    ENV = DNA_Env_Whole_Seq
    MODEL_NAME = "DNA_correction_whole"
elif MODE ==4:
    ENV = DNA_Error_Detection_Env_Without_BERT
    MODEL_NAME = "DNA_detection_without_BERT"
elif MODE == 5:
    ENV = DNA_Error_Detection_Single2_Env
    MODEL_NAME = "DNA_detection_single"




KMER_SHIFT = 0 # -1 = First base of triplet / 0 = middle base of triplet / 1 = last base of triplet
MODEL_NAME += "_" + str(KMER_SHIFT) +"_0.000001"

MODEL_PATH = "/Users/I570101/Documents/Bachelor-Thesis/DNA_RL/models/" + BERT_MODEL + '/' + MODEL.__name__ + '/ErrorRate' + str(ERROR_RATE)
MODEL_PATH += '/' + MODEL_NAME 

DNA_PATH = "/Users/I570101/Documents/Bachelor-Thesis/DNA_RL/seq1.txt"

print(MODEL_PATH)


def train(model, no_of_steps):
    for counter in range(0,no_of_steps):
        model.learning_starts = 10000
        model.learn(total_timesteps=10000)
        model.save(MODEL_PATH)

def predict(model, env):

    obs = env.reset()

    while True:
        action, _states = model.predict(obs)

        obs, rewards, dones, info = env.step(action)

        if (dones):
            break
    
def main():
    seq = open(DNA_PATH, "r").read().replace('\n', '').upper()
    use_bert_states = False
    if BERT_MODEL == "BERT": use_bert_states = True
    env = ENV(seq, random_processing=True, error_rate=ERROR_RATE, use_bert_states=use_bert_states, kmer_shift=KMER_SHIFT, seq_len=SAMPLE_SIZE)
    lr = LEARNING_RATE
    model = MODEL(SB3_POLICY, env, verbose=1,learning_rate = lr, device='cuda'
        # n_steps = 2048,
        # batch_size = 2048,
        # n_epochs=1,
        #use_sde=True
    )
    try:
        custom_objects = { 'learning_rate': lr}
        model = MODEL.load(MODEL_PATH, env, custom_objects=custom_objects)
    except Exception as e: print(e)

    train(model, 10000)

    predict(model, env)





if __name__ == "__main__":
    main()