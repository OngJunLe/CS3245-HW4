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
from pandas import Series


# Format of term dictionary {term: (offset, no_bytes), ...}
# Format of postings [(docID, log term frequency), ...]

# Unique keys for document length dictionary and total number of documents
DOCUMENT_LENGTH_KEY = -100
TOTAL_DOCUMENTS_KEY = -200

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d temp_postings-file -p postings-file")

def tokenize(input, stemmer, stopwords):
    temp_postings = defaultdict(list) 
    bigram_list = []
    tokens = word_tokenize(input)
    first = tokens.pop(0)
    first = first.split("D0C_ID")
    id = first[0]

    #Retain position of original word 
    tokens.insert(0, first[1])

    #Remove tokens with punctuation 
    tokens = [word for word in tokens if not any(char in string.punctuation for char in word)]

    #Remove stop words: common words that do not contribute to meaning of text
    tokens = [word for word in tokens if word.lower not in stopwords]

    #Stemming
    tokens = [stemmer.stem(word) for word in tokens] 

    for i in range(len(tokens) - 2):
        bigram = (tokens[i], tokens[i + 1])
        bigram_list.append(bigram) 
    bigram_list.extend(tokens)  
    bigram_list = Counter(bigram_list)

    #Create tuples of (docID, log term freqeuncy) and appending to temp_postings
    for word in bigram_list:
            temp_postings[word].append((id, 1 + math.log10(bigram_list[word])))
    return temp_postings

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
    stemmer = nltk.stem.PorterStemmer()
    total_documents = 0
    bigram_list = []
    tokens = []

    stoplist = set(stopwords.words('english'))
    with open(in_dir, encoding="utf-8") as f:
        df = pd.read_csv(f, sep=',', header=0, quotechar='"', quoting=csv.QUOTE_ALL)
        total_documents = len(df.index)
        bag = db.from_sequence(df['document_id'].apply(str) + "D0C_ID" + df['content'])
        token_counter_list = bag.map(tokenize, stemmer=stemmer, stopwords=stoplist).compute() 
        for dictionary in token_counter_list:
            for key in dictionary:
                #list of tuples instead of list of list of tuples
                temp_postings[key].append(dictionary[key][0])
            dictionary.clear()  
        '''
        # Document vector length calculation - removed normalisation for now
        sum = 0
        for word in bigram_list:
            sum += (1 + math.log10(bigram_list[word]))**2
        document_length = sum**0.5
        doc_length_dictionary[id] = document_length
        '''
    #sorted_keys = sorted(list(temp_postings.keys())) -- do the keys need to be sorted for some reason? cant rmb, double check since cant sort tuples and str
    # Storing byte offset in dictionary so that postings lists can be retrieved without reading entire file
    current_offset = 0 
    with open(out_postings, "wb") as output:
        '''
        # Add document length dictionary - removed normalisation for now 
        dictionary_binary = pickle.dumps(doc_length_dictionary)
        no_of_bytes = len(dictionary_binary)
        term_dictionary[DOCUMENT_LENGTH_KEY] = (current_offset, no_of_bytes)
        output.write(dictionary_binary)
        current_offset += no_of_bytes
        '''
        for key in temp_postings.keys():
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
  

'''
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
    #build_index("dataset.csv", "dictionary.txt", "postings.txt") 
    #build_index("test.csv", "test_dictionary.txt", "test_postings.txt") 


  



