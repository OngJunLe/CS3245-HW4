#!/usr/bin/python3
import nltk
import sys
import getopt
import pickle
import math
import csv
import pandas as pd
import dask.bag as db
import zlib
import struct
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

def is_int(word):
    try:
        int(word)
    except ValueError:
        return False
    else:
        return True
    
def tokenize(input, id, stemmer,stopwords, bigrams=False):
    temp_postings = defaultdict(list) 
    bigram_list = []
    tokens = word_tokenize(input)
    id = int(id)

    # keep only letters in each string
    tokens = [''.join([char for char in t if char.isalpha()]) for t in tokens]

    # remove empty tokens
    tokens = [t for t in tokens if t]

    #Remove stop words: common words that do not contribute to meaning of text
    tokens = [word for word in tokens if word.lower() not in stopwords]

    #Stemming
    tokens = [stemmer.stem(word) for word in tokens] 

    if bigrams:

        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            bigram_list.append(bigram) 
        bigram_list.extend(tokens)  
        bigram_list = Counter(bigram_list)

        #Create tuples of (docID, log term freqeuncy) and appending to temp_postings
        for word in bigram_list:
                temp_postings[word].append((id, 1 + math.log10(bigram_list[word])))
        return temp_postings
    else:
        tokens_count = Counter(tokens)
        for word in tokens_count:
            temp_postings[word].append((id, 1 + math.log10(tokens_count[word])))
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

    stoplist = set(stopwords.words('english'))
    with open(in_dir, encoding="utf-8") as f:
        df = pd.read_csv(f, sep=',', header=0, quotechar='"', quoting=csv.QUOTE_ALL)
        total_documents = len(df.index)

        # create dask bags to parallelize processing
        # map the tokenize function to these bags
        bag = db.from_sequence(df['content'])
        id_bag = db.from_sequence(df['document_id'].apply(str))
        token_counter_list = db.map(tokenize, bag, id_bag, stemmer=stemmer, stopwords=stoplist, bigrams=True).compute()

        bag2 = db.from_sequence(df['title'])
        token_counter_list2 = db.map(tokenize, bag2, id_bag, stemmer=stemmer, stopwords=stoplist).compute()

        bag3 = db.from_sequence(df['court'])
        token_counter_list3 = db.map(tokenize, bag3, id_bag, stemmer=stemmer, stopwords=stoplist).compute()

        # bag4 = db.from_sequence(df['date_posted'])

        # process the results of the dask processing
        # combine all the dictionaries into 1 main one
        for dictionary in token_counter_list:
            for key in dictionary:
                #list of tuples instead of list of list of tuples
                temp_postings[key].append(dictionary[key][0])
                
            dictionary.clear() 

        for dictionary in token_counter_list2:
            for key in dictionary:
                #list of tuples instead of list of list of tuples
                temp_postings["title#" + key].append(dictionary[key][0])
            dictionary.clear() 

        for dictionary in token_counter_list3:
            for key in dictionary:
                #list of tuples instead of list of list of tuples
                temp_postings["court#" + key].append(dictionary[key][0])
            dictionary.clear()  
        '''
        # Document vector length calculation - removed normalisation for now
        sum = 0
        for word in bigram_list:
            sum += (1 + math.log10(bigram_list[word]))**2
        document_length = sum**0.5
        doc_length_dictionary[id] = document_length
        '''
    
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

            # pack data as binary
            to_add_binary = b''.join(struct.pack('if', *tup) for tup in to_add)

            # compress binary stream
            to_add_binary = zlib.compress(to_add_binary)

            # write to file while storing offset
            no_of_bytes = len(to_add_binary)
            term_dictionary[key] = (current_offset, no_of_bytes)
            output.write(to_add_binary)
            current_offset += no_of_bytes

    # Add total number of documents to dictionary
    term_dictionary[TOTAL_DOCUMENTS_KEY] = total_documents
    dictionary_binary = pickle.dumps(term_dictionary)
    with open(out_dict, "wb") as output:
        output.write(dictionary_binary)
    
    # Not sure if should have seperate code file for this 
    # process_legal_dict("burtons_thesaurus.txt", stemmer, stoplist)

    print ("indexing over")



# process the raw .txt of the legal thesaurus we are using
def process_legal_dict(dictionary, stemmer, stoplist):
    legal_dict = {}
    with open(dictionary, "r", encoding="utf-8") as input:
        while True:
            line = input.readline()
            if ("LASTLINE" in line):
                break
            if (len(line) > 2 and "ASSOCIATED CONCEPTS" not in line and "FOREIGN PHRASES" not in line and "Generally" not in line and "Specifically" not in line):
                line = line.split(", ")
                term = stemmer.stem(line.pop(0).lower())
                del line[0]
                tokens = [word for word in line if len(word.split(" ")) < 3]
                # remove stopwords and stem
                tokens = [word for word in tokens if word.lower() not in stoplist]
                tokens = [stemmer.stem(word) for word in tokens]
                legal_dict[term] = tokens
    with open("binary_thesaurus.txt", "wb") as output:
        legal_dictionary_binary = pickle.dumps(legal_dict)
        output.write(legal_dictionary_binary)
    

# So this doesn't run when this file is imported in other scripts
if (__name__ == "__main__"): 

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


  



