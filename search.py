#!/usr/bin/python3
import re
import nltk
import sys
import getopt
from collections import defaultdict
from math import log
import heapq
import pickle
from query_processor import QueryProcessor
from nltk.corpus import wordnet as wn
# python search.py -d dictionary.txt -p postings.txt -q sanity-queries.txt -o output.txt

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')
    # This is an empty method
    # Pls implement your code in below

    with open(queries_file, 'r') as f:
        content = f.read().splitlines()  
        query = content.pop(0)
        relevant_docs = content

    qp = QueryProcessor(dict_file, postings_file)
    
    result_strings = []

    if ("AND" in query):
        result = qp.process_query_boolean(query)

    else:
        result = qp.process_query(query, number_results=10).split(" ")
    
    for doc in relevant_docs:
        if doc in result:
            #sort relevant docs by order in result
            result_strings.append((result.index(doc), doc))
            result.remove(doc)
        result_strings = sorted(result_strings)
        result_strings = [tuple[1] for tuple in result_strings]
    result_strings.extend(result)
    
    # boolean result string somehow not getting joined , double check
    with open(results_file, 'w') as f:
        f.write(' '.join(result_strings))
    print(f'output written to {results_file}')


if __name__ == "__main__":

    dictionary_file = postings_file = file_of_queries = output_file_of_results = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-d':
            dictionary_file  = a
        elif o == '-p':
            postings_file = a
        elif o == '-q':
            file_of_queries = a
        elif o == '-o':
            file_of_output = a
        else:
            assert False, "unhandled option"

    if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
        usage()
        sys.exit(2)

    run_search(dictionary_file, postings_file, file_of_queries, file_of_output)

# python search.py -d 'data/struct_compress_dictionary' -p 'data/struct_compress_postings' -o 'data/q1_output.txt'

