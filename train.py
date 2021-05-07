from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import CSVLogger
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing import text
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, BatchNormalization, GRU, concatenate, \
    Bidirectional
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import unicodedata
import unidecode
import base64
import nltk
import gensim
import re
import pickle
import pandas as pd
import numpy as np
import logging
from datetime import datetime
logging.basicConfig(level=logging.DEBUG, filename='./logs/logfile_training.txt')
csv_logger = CSVLogger('logfile_training.txt', append=True)

logging.info('')
logging.info('Model Training - Phase - Started at {}'.format(datetime.now()))
logging.info('----------------------')
logging.info('')

MAX_SEQ_LENGTH = 500
TOP_N_TAGS = 500
W2V_SIZE = MAX_SEQ_LENGTH
W2V_WINDOW = 7
W2V_MIN_COUNT = 10
W2V_EPOCH = 32


# Define tensorflow keras based neural network model
def create_model(vocab_size, label_count, embedding_matrix, kernel_initializer, dropout_rate):
    model = Sequential()
    model.add(Embedding(vocab_size + 1, W2V_SIZE, weights=[embedding_matrix], input_length=MAX_SEQ_LENGTH,
                        trainable=False))
    # model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
    # model.add(Bidirectional(LSTM(units=256, return_sequences=True)))
    model.add(LSTM(500))
    model.add(Dense(units=10000,
                    kernel_initializer=kernel_initializer,
                    activation='relu'
                    )
              )
    model.add(Dropout(dropout_rate[0]))
    model.add(Dense(units=1250,
                    kernel_initializer=kernel_initializer,
                    activation='relu'
                    )
              )
    model.add(Dropout(dropout_rate[1]))
    model.add(Dense(units=750,
                    kernel_initializer=kernel_initializer,
                    activation='relu'
                    )
              )
    model.add(Dense(label_count, activation='sigmoid'))

    #logging.info(model.summary())
    # Open the file
    with open('logfile.txt', 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    logging.info('')

    return model


# vocab_size = 12913
# vocab_size = 232822

with open('models/embedding_matrix.pickle', 'rb') as handle:
    embedding_matrix = pickle.load(handle)

with open('models/vocab_size.pickle', 'rb') as handle:
    vocab_size = pickle.load(handle)

# Create tensorflow keras based neural network model
model = create_model(vocab_size, TOP_N_TAGS, embedding_matrix, 'glorot_uniform', [0.01, 0.02])

# Compile the model with learning parameters
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

# model.compile(loss='sparse_categorical_crossentropy', metrics=['acc'])

# Define Early stopping strategy to handle model over-fitting problem
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
             EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]

# Define the batch size
BATCH_SIZE = 256

with open('models/X_train_padded.pickle', 'rb') as handle:
    X_train_padded = pickle.load(handle)

with open('models/y_train.pickle', 'rb') as handle:
    y_train = pickle.load(handle)


# train the tensorlfow keras model
history = model.fit(X_train_padded, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=10,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=callbacks)
                    #callbacks=[csv_logger])
logging.info("Train Log: ")
#logging.info(callbacks)+

# Save the trained model
model.save('models/tag_predictor_keras_model.h5')
logging.info("Trained model has been saved as 'tag_predictor_keras_model.h5'")
logging.info('')
logging.info('Model Training - Phase - Completed at {}'.format(datetime.now()))
logging.info('')
