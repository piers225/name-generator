import csv

def csv_reader():

    with open('names.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader) 
        names = [name for row in reader for name in row]

    corpus = ' '.join(names).lower()

    chars = sorted(set(corpus))

    char_to_index = {char: index for index, char in enumerate(chars)}
    index_to_char = {index: char for index, char in enumerate(chars)}

    return chars, char_to_index, index_to_char, corpus