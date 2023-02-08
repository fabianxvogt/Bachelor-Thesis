from data.DNA_sequence_manager import DNA_sequence_manager
from utilities.dna_utils import distort_seq
from utilities.bert_utils import generate_dnabert_states
import pickle
from keras import Sequential
from keras.layers.convolutional.conv1d import Conv1D
from keras.layers.pooling.max_pooling1d import MaxPooling1D
from keras.layers.reshaping.flatten import Flatten
from keras.layers.core.dense import Dense
from sklearn.metrics import accuracy_score

import numpy as np

def train(model, X_train, y_train):

    # Train the model
    y_train_2class = [[0,0] for y in y_train]
    for i, y in enumerate(y_train): y_train_2class[i][y] = 1 
    model.fit(X_train, np.array(y_train))



def predict(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_pred_abs = [np.argmax(y_pred[i]) for i in range(0, len(y_pred))]

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred_abs)
    print(f'Accuracy: {accuracy:.2f}')


seq_len = 500
DNA_seq_manager = DNA_sequence_manager(seq_len)



num_classes = 2

model = Sequential()
model.add(Conv1D(64, 2, activation="relu", input_shape=(768,1)))
model.add(Dense(16, activation="relu"))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(2, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy',
     optimizer = "adam",               
              metrics = ['accuracy'])
model.summary()

for i in range(0, 10000):

    #df_train, df_test = prepare_data(random_symbol, timeframe)

    dna_sample = DNA_seq_manager.get_new_sequence().upper()
    error_seq, error_map = distort_seq(dna_sample, 0.1)
    bert_states = generate_dnabert_states(error_seq, False)

    dna_sample = dna_sample[1:-1]
    error_seq = error_seq[1:-1]
    error_map = error_map[1:-1]


    X, y = np.array(bert_states.detach()), np.array(error_map) 

    # Save the model to a file
    if i % 10 == 0 and i > 0:
        predict(model, X, y)
        with open('cnn_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            f.close()
    else:
        train(model, X, y)