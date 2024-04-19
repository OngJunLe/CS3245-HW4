#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import string
import pickle
import os
import collections
import math
import csv
import pandas as pd
import dask.bag as db

from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from collections import defaultdict


# Format of term dictionary {term: (offset, no_bytes), ...}
# Format of postings [(docID, log term frequency), ...]

# Unique keys for document length dictionary and total number of documents
DOCUMENT_LENGTH_KEY = -100
TOTAL_DOCUMENTS_KEY = -200

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d temp_postings-file -p postings-file")

def tokenize(query, stemmer, stopwords):
    tokens = word_tokenize(query)

    #Remove tokens with punctuation 
    tokens = [word for word in tokens if not any(char in string.punctuation for char in word)]

    #Remove stop words: common words that do not contribute to meaning of text
    tokens = [word for word in tokens if word.lower not in stopwords]

    #Stemming
    tokens = [stemmer.stem(word) for word in tokens] 

    return tokens

def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the temp_postings file and postings file
    """
    print('indexing...')
    # This is an empty method
    # Pls implement your code in below
    temp_postings = defaultdict(list) 
    term_dictionary = {}
    doc_length_dictionary = {}
    stemmer = nltk.stem.PorterStemmer()
    total_documents = 0
    maxInt = sys.maxsize
    bigram_list = []
    tokens = []

    stoplist = set(stopwords.words('english'))
    with open(in_dir, encoding="utf-8") as f:
        df = pd.read_csv(f, sep=',', header=0, quotechar='"', quoting=csv.QUOTE_ALL)
        bag = db.from_sequence(df['content'])
        token_counter_list = bag.map(tokenize, stemmer=stemmer, stopwords=stoplist).compute() 
        for token_list in token_counter_list:
             for word in token_list:
                  tokens.append(word)
        for i in range(len(tokens) - 2):
                    bigram = (tokens[i], tokens[i + 1])
                    bigram_list.append(bigram)
        bigram_list = Counter(bigram_list)
        # Create tuples of (docID, log term freqeuncy) and appending to temp_postings
        for word in bigram_list:
            temp_postings[word].append((id, 1 + math.log10(bigram_list[word])))
        # Document vector length calculation 
        sum = 0
        for word in bigram_list:
            sum += (1 + math.log10(bigram_list[word]))**2
        document_length = sum**0.5
        doc_length_dictionary[id] = document_length
    sorted_keys = sorted(list(temp_postings.keys()))   
    print(sorted_keys)
    # Storing byte offset in dictionary so that postings lists can be retrieved without reading entire file
    current_offset = 0 
    with open(out_postings, "wb") as output:
        # Add document length dictionary
        dictionary_binary = pickle.dumps(doc_length_dictionary)
        no_of_bytes = len(dictionary_binary)
        term_dictionary[DOCUMENT_LENGTH_KEY] = (current_offset, no_of_bytes)
        output.write(dictionary_binary)
        current_offset += no_of_bytes
        for key in sorted_keys:
            to_add = sorted(temp_postings[key])
            to_add_binary = pickle.dumps(to_add)
            no_of_bytes = len(to_add_binary)
            term_dictionary[key] = (current_offset, no_of_bytes)
            output.write(to_add_binary)
            current_offset += no_of_bytes

    # Add total number of documents to dictionary
    term_dictionary[TOTAL_DOCUMENTS_KEY] = total_documents
    dictionary_binary = pickle.dumps(term_dictionary)
    with open(out_dict, "wb") as output:
        output.write(dictionary_binary)
    
    print ("indexing over")

#build_index("dataset.csv", "dictionary.txt", "postings.txt")   

'''
build_index("dataset.csv", "dictionary.txt", "postings.txt")  
with open("dictionary.txt", "rb") as input:
    dictionary = pickle.loads(input.read())
    offset, to_read = dictionary["court"] 
with open("postings.txt", "rb") as input:
    input.seek(offset)
    postings = pickle.loads(input.read(to_read))
    print(postings)
'''


# So this doesn't run when this file is imported in other scripts
if (__name__ == "__main__"): 
    '''
    input_directory = output_file_dictionary = output_file_postings = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-i': # input directory
            input_directory = a
        elif o == '-d': # temp_postings file
            output_file_dictionary = a
        elif o == '-p': # postings file
            output_file_postings = a
        else:
            assert False, "unhandled option"

    if input_directory == None or output_file_postings == None or output_file_dictionary == None:
        usage()
        sys.exit(2)

    build_index(input_directory, output_file_dictionary, output_file_postings)   
    '''
    build_index("test.csv", "dictionary.txt", "postings.txt")  


  



