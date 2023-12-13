import numpy as np
from src.csv_reader import csv_reader

def generate_name(model, seed_text, length=10):

    chars, char_to_index, index_to_char, _ = csv_reader()

    generated_name = seed_text

    for _ in range(length):
        seed_seq = [char_to_index[char] for char in seed_text]
        seed_seq = np.array(seed_seq).reshape(1, -1)
        
        predicted_index = np.argmax(model.predict(seed_seq), axis=-1)
        predicted_char = index_to_char[predicted_index[0]]
        
        generated_name += predicted_char
        seed_text = seed_text[1:] + predicted_char

    return generated_name
