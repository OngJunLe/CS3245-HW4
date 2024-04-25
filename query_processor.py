import pickle
import heapq
import re
from collections import defaultdict
from math import log
#from index import TOTAL_DOCUMENTS_KEY
from nltk import PorterStemmer, word_tokenize
from string import punctuation
from nltk.corpus import wordnet as wn
import zlib
import struct


TOTAL_DOCUMENTS_KEY = -200
class QueryProcessor:
    OPERATOR_AND = 2
    OPERATOR_OR = 3
    OPERATOR_LIST = [OPERATOR_AND, OPERATOR_OR]

    def __init__(self, dictionary_file, postings_file):
        with open(dictionary_file, 'rb') as f:
            self.dictionary = pickle.load(f)

        with open(postings_file, 'rb') as f:
            '''
            #Removed normalization for now 
            offset, bytes_to_read = self.dictionary[DOCUMENT_LENGTH_KEY]
            f.seek(offset)
            self.document_length_dictionary = pickle.loads(f.read(bytes_to_read))
            '''

        self.postings_file = postings_file

        self.stemmer = PorterStemmer()

    
    def process_query(self, query, number_results=10):
        to_add = []
        scores = defaultdict(float)
        

        # tokenize 
        query_terms = word_tokenize(query)
        
        # invalid free text query if it contains quotations
        if ('``' in query_terms):
            return ""
        
        # remove punctuation
        # query_terms = [term for term in query_terms if term not in punctuation]

        # keep only letters in each string
        query_terms = [''.join([char for char in t if char.isalpha()]) for t in query_terms]

        # remove empty tokens
        query_terms = [t for t in query_terms if t]

        # remove duplicate terms
        query_terms = set(query_terms)
        
        # stem the terms
        query_terms = [self.stemmer.stem(term.lower()) for term in query_terms]

        terms_with_fields = []
        for term in query_terms:
            terms_with_fields.append("court#" + term)
            terms_with_fields.append("title#" + term)

        # add additional related query terms from legal thesaurus

        with open("binary_thesaurus.txt", "rb") as input:
            thesaurus = pickle.loads(input.read())
            for term in query_terms:
                if term in thesaurus.keys():
                    to_add.extend(thesaurus[term])

        query_terms.extend(to_add)

        # remove invalid terms
        query_terms = [term for term in query_terms if term in self.dictionary]
        terms_with_fields = [term for term in terms_with_fields if term in self.dictionary]
        
        log_N = log(self.dictionary[TOTAL_DOCUMENTS_KEY])

        for term in query_terms:
            postings_list = self.fetch_postings_list(term)

            # TFIDF ALREADY CALCULATED IN FETCH_POSTINGS_LIST
            # calculate tf.idf for this term
            # heuristic: assume log freq weight of term t in query = 1
            # docu_freq = len(postings_list)
            # weight_term = log_N - log(docu_freq)

            for (doc_id, weight_docu) in postings_list:
                # compute tf.idf for term in document
                # would idf for queries just involve multiplying this by weight term again? since idf is same, tf should be 1 in query since we remove duplicates 
                scores[doc_id] += weight_docu

        for term in terms_with_fields:
            postings_list = self.fetch_postings_list(term, extra_weight=3)

            for (doc_id, weight_docu) in postings_list:
                # compute tf.idf for term in document
                # would idf for queries just involve multiplying this by weight term again? since idf is same, tf should be 1 in query since we remove duplicates 
                scores[doc_id] += weight_docu

        if len(scores) == 0:
            return ""

        result = sorted(scores.items(), key=lambda item: item[1])
        ids_to_return = [str(item[0]) for item in result]

        return " ".join(ids_to_return)
    

    def fetch_postings_list(self, term, extra_weight=1):
        if term not in self.dictionary.keys():
            return []

        log_N = log(self.dictionary[TOTAL_DOCUMENTS_KEY])
        offset, bytes_to_read = self.dictionary[term]

        # read the postings list from the postings file
        with open(self.postings_file, 'rb') as f:
            f.seek(offset)

            read_data = f.read(bytes_to_read)

            # Decompress and unpack data
            decompressed_data = zlib.decompress(read_data)
            postings_list = [struct.unpack('if', decompressed_data[i:i+8]) for i in range(0, len(decompressed_data), 8)]

            # calculate tf.idf before returning postings: postings currently tuples of (docID, tf)
            # heuristic: assume log freq weight of term t in query = 1
            docu_freq = len(postings_list)
            weight_term = log_N - log(docu_freq)
            
            postings_list = [(tup[0], tup[1] * weight_term * extra_weight) for tup in postings_list]
  
        return postings_list
    
    def process_query_boolean(self, query):
        tokens = word_tokenize(query)

        #remove punctuation 
        # tokens = [term for term in tokens if term not in punctuation]

        # keep only letters in each string
        tokens = [''.join([char for char in t if char.isalpha()]) for t in tokens]

        # remove empty tokens
        tokens = [t for t in tokens if t]

        #remove quotation marks - double check why theres 2 unique quotation marks strings
        tokens = [term for term in tokens if term != "``"]
        tokens = [term for term in tokens if term != "''"]
        tokens = [self.OPERATOR_AND if term == "AND" else term for term in tokens]

        # stem and lower terms
        tokens = [self.stemmer.stem(term.lower()) if term != self.OPERATOR_AND else term for term in tokens]

        ## convert terms to bigrams

        # split the tokens by AND, each item between ANDs becomes a list of token(s)
        items = []
        prev_item = []
        for t in tokens:
            if t == 2:
                items.append(prev_item)
                prev_item = []
            else:
                prev_item.append(t)
        items.append(prev_item)

        # convert items with length 3 or 2 to bigrams
        final_items = []
        for i,item in enumerate(items):
            if len(item)>3 or len(item) == 0:
                raise ValueError("Boolean query contains invalid item")
            # if length is 3, create 2 bigrams and 3 single terms
            if len(item) == 3:
                final_items.append((item[0], item[1]))
                final_items.append(self.OPERATOR_OR)
                final_items.append((item[1], item[2]))
                final_items.append(self.OPERATOR_OR)
                final_items.append(item[0])
                final_items.append(self.OPERATOR_OR)
                final_items.append(item[1])
                final_items.append(self.OPERATOR_OR)
                final_items.append(item[2])
                if i < len(items)-1:
                    final_items.append(self.OPERATOR_AND)
            # if length is 2, create 1 bigram and 2 single terms
            elif len(item) == 2:
                final_items.append((item[0], item[1]))
                final_items.append(self.OPERATOR_OR)
                final_items.append(item[0])
                final_items.append(self.OPERATOR_OR)
                final_items.append(item[1])
                if i < len(items)-1:
                    final_items.append(self.OPERATOR_AND)
            # if length is 1, just add the single term
            elif len(item) == 1:
                final_items.append(item[0])
                if i < len(items)-1:
                    final_items.append(self.OPERATOR_AND)


        # put this here as token list might become less than 3 tokens after bigraming e.g. (term1, term2) AND 
        if len(final_items) < 3: 
            return ""

        # convert to postfix notation
        postfix = self.convert_to_postfix(final_items)

        # evaluate query
        result = self.evaluate_postfix(postfix)

        # sort tuples of (docID, tf-idf) by tf-idf
        result = sorted(result, key = lambda x: x[1], reverse = True)

        #only return docIDs
        result = [str(r[0]) for r in result]

        # STOP GAP MEASURE TO ACCOUNT FOR DUPLICATES BEING RETURNED
        # had a problem with the framework detecting duplicates
        # we think we fixed the bug that was causing this, but left this in in case (not enough runs left to troubleshoot)
        seen = set()
        final_result = []
        for item in result:
            if item not in seen:
                seen.add(item)
                result.append(item)
            else:
                print(f"already seen {item}, not adding to list")
        
        return final_result

    # use shunting-yard algorithm to process query into postfix notation
    def convert_to_postfix(self, tokens):
        output_queue = []
        operator_stack = []

        for token in reversed(tokens):
            if token in self.OPERATOR_LIST:
                while (operator_stack and operator_stack[-1] >= token):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            else: # token is a term
                output_queue.append(token)

        while operator_stack:
            output_queue.append(operator_stack.pop())

        return output_queue
    
    # evaluate the query
    def evaluate_postfix(self, postfix):
        eval_stack = []
        
        for token in postfix:
            if token in self.OPERATOR_LIST:
                if token == self.OPERATOR_OR:
                    eval_stack.append(self.or_operation(eval_stack.pop(), eval_stack.pop()))

                elif token == self.OPERATOR_AND:
                    eval_stack.append(self.and_operation(eval_stack.pop(), eval_stack.pop()))
            else:
                # if it's a bigram, directly add
                if isinstance(token, tuple):
                    eval_stack.append(self.fetch_postings_list(token))
                else:
                    # if it's a single word, try to check if it matches any field tokens
                    if "court#"+token in self.dictionary.keys():
                        eval_stack.append(self.fetch_postings_list("court#"+token))
                    elif "title#"+token in self.dictionary.keys():
                        eval_stack.append(self.fetch_postings_list("title#"+token))
                    else:
                        eval_stack.append(self.fetch_postings_list(token))
                
        return eval_stack[0]


    def and_operation(self, postings1, postings2):
        #current index for each postings 
        current_index_1 = 0
        current_index_2 = 0

        #calculate skip length for each postings
        skip_1 = int(len(postings1) ** 0.5)
        skip_2 = int(len(postings2) ** 0.5)
        results_list = []

        # while current index still in bounds
        while current_index_1 < len(postings1) and current_index_2 < len(postings2):
            # if current docIDs match
            if (postings1[current_index_1][0] == postings2[current_index_2][0]):
                to_add = postings1[current_index_1]
                to_add_2 = postings2[current_index_2]

                # add tf-idf for both terms together
                # to_add[0] is just the docID, which will be same for both postings
                to_add = (to_add[0], to_add[1] + to_add_2[1])

                results_list.append(to_add)
                current_index_1 += 1
                current_index_2 += 1

            # if current docID in postings 1 is smaller than postings2
            elif (postings1[current_index_1][0] < postings2[current_index_2][0]):
                # if skip pointer not out of bounds, add skip to current index in postings1 
                # else just iterate to next node 
                if (current_index_1 + skip_1 < len(postings1)):
                    if (postings1[current_index_1 + skip_1][0] <= postings2[current_index_2][0]):
                        current_index_1 += skip_1
                    else:
                        current_index_1 += 1
                else:
                    current_index_1 += 1
            # if current docID in postings2 is smaller than postings1
            # same logic as above
            else:
                if (current_index_2 + skip_2 < len(postings2)):
                    if (postings2[current_index_2 + skip_2][0] <= postings1[current_index_1][0]):
                        current_index_2 += skip_2
                    else:
                       current_index_2 += 1
                else:
                    current_index_2 += 1
        
        return results_list

    def or_operation(self, postings1, postings2):
        #current index for each postings 
        current_index_1 = 0
        current_index_2 = 0

        results_list = []

        # while current index still in bounds
        while current_index_1 < len(postings1) and current_index_2 < len(postings2):
            # if current docIDs match
            if (postings1[current_index_1][0] == postings2[current_index_2][0]):

                # add tf-idf for both terms together
                # first item is just the docID, which will be same for both postings
                to_add = (postings1[current_index_1][0], postings1[current_index_1][1] + postings2[current_index_2][1])

                results_list.append(to_add)
                current_index_1 += 1
                current_index_2 += 1

            # if current docID in postings 1 is smaller than postings2
            elif (postings1[current_index_1][0] < postings2[current_index_2][0]):
                results_list.append(postings1[current_index_1])
                current_index_1 += 1
            # if current docID in postings2 is smaller than postings1
            # same logic as above
            else:
                results_list.append(postings2[current_index_2])
                current_index_2 += 1
        
        # add remaining parts of lists
        if current_index_1<len(postings1):
            results_list.extend(postings1[current_index_1:])
        if current_index_2<len(postings2):
            results_list.extend(postings2[current_index_2:])
        
        return results_list


'''
if __name__ == "__main__":
    qp = QueryProcessor("data/struct_compress_dictionary", "data/struct_compress_postings")
    query = 'quiet phone call'

    print(qp.process_query(query))
'''