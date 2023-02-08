import pickle
import random
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data.DNA_sequence_manager import DNA_sequence_manager
from utilities.dna_utils import distort_seq
from utilities.bert_utils import generate_dnabert_states

def train(model, X_train, y_train):

    # Train the model
    model.fit(X_train, y_train)



def predict(model, X_test, y_test):

    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')


# Create a decision tree model
load = False
if load == True:
    # Load the model from the file
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
else:
    model = DecisionTreeClassifier()
DNA_seq_manager = DNA_sequence_manager(500)
for i in range(0, 10000):

    #df_train, df_test = prepare_data(random_symbol, timeframe)

    dna_sample = DNA_seq_manager.get_new_sequence().upper()
    error_seq, error_map = distort_seq(dna_sample, 0.1)
    bert_states = generate_dnabert_states(error_seq, False)

    dna_sample = dna_sample[1:-1]
    error_seq = error_seq[1:-1]
    error_map = error_map[1:-1]
    

    # Save the model to a file
    if i % 10 == 0 and i > 0:
        predict(model, np.array(bert_states.detach()), np.array(error_map))
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
            f.close()
    else:
        train(model, np.array(bert_states.detach()), np.array(error_map))