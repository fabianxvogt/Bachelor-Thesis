# Reinforcement Learning for DNA-sequencing-errors

## Prerequisites

1. Download the pre-trained DNABERT-Model for KMER = 3: https://drive.google.com/file/d/1nVBaIoiJpnwQxiz4dSq6Sv9kBKfXhZuM/view
2. In DNA_RL/main.py, set the  ```BASE_PATH ``` to the full path of your DNA_RL folder: \
 ```BASE_PATH = "/Users/I570101/Documents/Bachelor-Thesis/DNA_RL/" ```

3. Modify other parameters to configure training: 

    3.1. ```BERT_MODEL```: The DNABERT-Model to use: Either ```BERT``` or ```BERT_LM```

    3.2. ```LEARNING_RATE```: The Learning-Rate of the RL-Agent 

    3.3. ```ERROR_RATE```: Rate of random errors in input-DNA 

    3.4. ```MODEL```: RL-Agent to use (out of the stable_baselines3 package), RecurrentPPO is recommended 

    3.5. ```KMER_SHIFT```: Defines the mapping from 3-KMER to a single DNA base: 

    ```-1```:   Map to first base of 3-KMER\
    ```0```:    Map to middle base of 3-KMER\
    ```+1```:   Map to last base of 3-KMER

    3.6. ```MODE``` Select from predefined training modes:

    ```1```: Error detection with multiple actions in a single run per sequence\
    ```2```: Error detection with a single actions per run in multiple runs per sequence\
    ```3```: Error correction with multiple actions in a single run per sequence\
    ```4```: Error detection in sequential processing\
    ```5```: Error detection with correction through DNABERT masking


## Run the RL-DNA-Correction project

1. Go to the DNA_RL folder \
 ```cd DNA_RL```
2. Activate the virtual env: \
 ```source env/bin/activate```
3. Install requirements \
 ```pip3 install -r requirements.txt```
4. Run the project \
 ```python main.py```
