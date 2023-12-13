import numpy as np
import tensorflow as tf
from src.csv_reader import csv_reader

def build_model():

    chars, char_to_index, index_to_char, corpus = csv_reader()

    num_data = [char_to_index[char] for char in corpus]

    seq_length = 3
    sequences = []
    targets = []

    for i in range(len(num_data) - seq_length):
        seq = num_data[i:i + seq_length]
        target = num_data[i + seq_length]
        sequences.append(seq)
        targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(chars), 8, input_length=seq_length),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(len(chars), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(sequences, targets, epochs=50)

    return model