import tensorflow as tf
import numpy as np

names = [
    "John", "Jane", "Bob", "Alice", "Charlie",
    "David", "Emma", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Kate", "Leo", "Mia",
    "Noah", "Olivia", "Peter", "Quinn", "Ryan",
    "Sophia", "Tom", "Ursula", "Vincent", "Wendy",
    "Xander", "Yvonne", "Zack", "Amelia", "Ben",
    "Cora", "Daniel", "Eva", "Felix", "Georgia",
    "Hugo", "Isabel", "Jason", "Kara", "Liam",
    "Mila", "Nathan", "Oscar", "Penelope", "Quincy",
    "Riley", "Stella", "Theo", "Uma", "Victor",
    "Willow", "Xavier", "Yasmine", "Zane", "Ava",
    "Bryce", "Clara", "Dylan", "Ella", "Finn",
    "Giselle", "Harrison", "Isla", "Jacob", "Kylie",
    "Lucas", "Madison", "Nora", "Owen", "Piper",
    "Quinn", "Rose", "Samuel", "Taylor", "Ulysses",
    "Violet", "Wyatt", "Xena", "Yara", "Zara",
    "Andrew", "Bella", "Christopher", "Daisy", "Edward",
    "Freya", "George", "Hazel", "Isaac", "Jasmine",
    "Kevin", "Lily", "Michael", "Natalie", "Oliver",
    "Paige", "Quentin", "Rachel", "Stephen", "Tara",
    "Ursula", "Vincent", "Wendy", "Xander", "Phil",
    "Zack", "John", "Max", "Martin", "Chris", "Steve",
    "Peter", "Dylan", "Kate", "Dave", "Scott"
]

corpus = ' '.join(names).lower()

chars = sorted(set(corpus))

char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}

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


def generate_name(model, seed_text, length=10):
    generated_name = seed_text

    for _ in range(length):
        seed_seq = [char_to_index[char] for char in seed_text]
        seed_seq = np.array(seed_seq).reshape(1, -1)
        
        predicted_index = np.argmax(model.predict(seed_seq), axis=-1)
        predicted_char = index_to_char[predicted_index[0]]
        
        generated_name += predicted_char
        seed_text = seed_text[1:] + predicted_char

    return generated_name


seed_text = "p".lower()
generated_name = generate_name(model, seed_text, length=5)

print(f"Generated Name: '{generated_name}'")