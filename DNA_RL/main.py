
from gym_envs.dna_error_detection_env import DNA_Error_Detection_Env
from gym_envs.nucleotide_wise_processing.single_run_with_multiple_actions.dna_error_detection_with_dnabert_masking_env import DNA_Error_Detection_With_Masking_Correction_Env
from gym_envs.nucleotide_wise_processing.single_run_with_multiple_actions.dna_error_detection_single_run_env import DNA_Error_Detection_Single_Run_Env
from gym_envs.nucleotide_wise_processing.single_action_with_multiple_runs.dna_error_detection_single_action_per_run import DNA_Error_Detection_Single_Action_Per_Run_Env

from gym_envs.nucleotide_wise_processing.single_run_with_multiple_actions.dna_error_correction_single_run_env import DNA_Error_Correction_Single_Run_Env
from gym_envs.sequential_processing.dna_error_detection_sequential_env import DNA_Error_Detection_Sequential_Env

from stable_baselines3 import A2C, DQN
#from stable_baselines3
from sb3_contrib import RecurrentPPO

from utilities.plot_manager import PlotManager

BASE_PATH = "/Users/I570101/Documents/Bachelor-Thesis/DNA_RL/"

LEARNING_RATE = 0.0001

BERT_ENCODE_MODEL = "BERT"
BERT_LANG_MODEL = "BERT_LM"
BERT_MODEL = BERT_ENCODE_MODEL

ENV = None
ERROR_RATE = 0.1
SAMPLE_SIZE = 100
MODEL = RecurrentPPO
LSTM_MULTI_POLICY = "MultiInputLstmPolicy"
LSTM_POLICY = "MlpLstmPolicy"
SB3_POLICY = LSTM_POLICY

if MODEL == A2C:
    SB3_POLICY = "MlpPolicy"


KMER_SHIFT = 0 # -1 = First base of triplet / 0 = middle base of triplet / 1 = last base of triplet

MODE = 1

if MODE == 1: # detect multiple errors in a single iteration
    ENV = DNA_Error_Detection_Single_Run_Env
    MODEL_NAME = "DNA_detection_single_run"
elif MODE == 2: # detect one error per iteration for multiple runs
    ENV = DNA_Error_Detection_Single_Action_Per_Run_Env
    MODEL_NAME = "DNA_detection_multi_run"
elif MODE == 3:
    ENV = DNA_Error_Correction_Single_Run_Env
    MODEL_NAME = "DNA_correction_single_run"
elif MODE == 4:
    ENV = DNA_Error_Detection_Sequential_Env
    MODEL_NAME = "DNA_detection_sequentially"
elif MODE == 5: 
    ENV = DNA_Error_Detection_With_Masking_Correction_Env
    MODEL_NAME = "DNA_masking_correction"


MODEL_NAME += "_" + str(KMER_SHIFT)# +"_0.000001"

MODEL_PATH = BASE_PATH + "model_data/" + BERT_MODEL + '/' + MODEL.__name__ + '/ErrorRate' + str(ERROR_RATE)
MODEL_PATH += '/' + MODEL_NAME 

DNA_PATH = BASE_PATH + "seq1.txt"

print(MODEL_PATH)


def train(model, env, no_of_steps, make_plot = True):
    if make_plot:
        plot_man = PlotManager()
        plot_man.load_historical_plot_data(MODEL_PATH + "_hist.npy")
        env.set_plot_data(
            (plot_man.errors_corrected_history or [0])[-1], 
            (plot_man.errors_found_history     or [0])[-1], 
            (plot_man.errors_missed_history    or [0])[-1], 
            (plot_man.errors_made_history      or [0])[-1], 
            (plot_man.corrects_found_history   or [0])[-1]
        )

    for counter in range(0,no_of_steps):
        model.learning_starts = 100
        model.learn(total_timesteps=1000)
        model.save(MODEL_PATH)
        
        if make_plot:
            plot_man.append_plot_data(env.get_plot_data())
            plot_man.save_historical_plot_data(MODEL_PATH + "_hist.npy")
            plot_man.create_plot_and_save(MODEL_PATH + "_fig1")

def predict(model, env):

    obs = env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        if (dones):
            break
    
def main():
    seq = open(DNA_PATH, "r").read().replace('\n', '').upper()
    use_lm = True
    if BERT_MODEL == "BERT": use_lm = False
    env = ENV(error_rate=ERROR_RATE, use_bert_for_masked_lm=use_lm, kmer_shift=KMER_SHIFT, seq_len=SAMPLE_SIZE)
    lr = LEARNING_RATE

    model = MODEL(SB3_POLICY, env, verbose=1,learning_rate = lr, device='cuda')
        # n_steps = 2048,
        # batch_size = 2048,
        # n_epochs=1,
        # use_sde=True
    
    try:
        custom_objects = { 'learning_rate': lr}
        model = MODEL.load(MODEL_PATH, env, custom_objects=custom_objects)
    except Exception as e: print(e)

    train(model, env, 10000)

    predict(model, env)


if __name__ == "__main__":
    main()