
import sys

import gym
from utilities.dna_utils import distort_seq
from utilities.bert_utils import bert_seq
from gym_envs.dna_error_correction_env import DNA_Environment
from gym_envs.dna_error_detection_env import DNA_Error_Detection_Env
from gym_envs.dna_env_whole_seq import DNA_Env_Whole_Seq
import random

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, DQN
from sb3_contrib import RecurrentPPO


BERT_MODEL = "BERT"
ENV = None
ERROR_RATE = 0.1
MODEL = RecurrentPPO
SB3_POLICY = "MlpLstmPolicy"

MODE = 1
if MODE == 1: # detect
    ENV = DNA_Error_Detection_Env
    MODEL_NAME = "DNA_detection"
elif MODE == 2: # correct
    ENV = DNA_Environment
    MODEL_NAME = "DNA_correction"
elif MODE == 3:
    ENV = DNA_Env_Whole_Seq
    MODEL_NAME = "DNA_correction_whole"


KMER_SHIFT = 0 # -1 = First base of triplet / 0 = middle base of triplet / 1 = last base of triplet
MODEL_NAME += "_" + str(KMER_SHIFT)

MODEL_PATH = "/Users/I570101/Documents/Bachelor-Thesis/DNA_RL/models/" + BERT_MODEL + '/' + MODEL.__name__ + '/ErrorRate' + str(ERROR_RATE)
MODEL_PATH += '/' + MODEL_NAME 

DNA_PATH = "/Users/I570101/Documents/Bachelor-Thesis/DNA_RL/seq1.txt"

print(MODEL_PATH)


def train(model, no_of_steps):
    for counter in range(0,no_of_steps):
        model.learning_starts = 10000
        model.learn(total_timesteps=100000)
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
    env = ENV(seq,ERROR_RATE, None, True,use_bert_states=use_bert_states, kmer_shift=KMER_SHIFT)
    lr = 0.0001
    model = MODEL(SB3_POLICY, env, verbose=1,learning_rate = lr)
    try:
        custom_objects = { 'learning_rate': lr}
        model = MODEL.load(MODEL_PATH, env, custom_objects=custom_objects)
    except Exception as e: print(e)

    train(model, 10000)

    predict(model, env)





if __name__ == "__main__":
    main()