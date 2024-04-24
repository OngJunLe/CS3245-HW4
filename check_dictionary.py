#!/usr/bin/python
import pickle

def check_dictionary():
    dict_file = 'data/dictionary'

    with open(dict_file, 'rb') as f:
        dictionary = pickle.load(f)

    log_file = 'data/terms.txt'
    with open(log_file, 'w') as f:
        for k in dictionary.keys():
            f.write(f"{k}\n")

if __name__ == "__main__":
    check_dictionary()