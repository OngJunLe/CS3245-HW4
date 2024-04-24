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
        query_terms = [term for term in query_terms if term not in punctuation]

        # remove duplicate terms
        query_terms = set(query_terms)
        
        # stem the terms
        query_terms = [self.stemmer.stem(term.lower()) for term in query_terms]

        # add additional related query terms from legal thesaurus

        with open("binary_thesaurus.txt", "rb") as input:
            thesaurus = pickle.loads(input.read())
            for term in query_terms:
                if term in thesaurus.keys():
                    to_add.extend(thesaurus[term])

        query_terms.extend(to_add)

        # remove invalid terms
        query_terms = [term for term in query_terms if term in self.dictionary]
        
        log_N = log(self.dictionary[TOTAL_DOCUMENTS_KEY])

        for term in query_terms:
            postings_list = self.fetch_postings_list(term)

            # calculate tf.idf for this term
            # heuristic: assume log freq weight of term t in query = 1
            docu_freq = len(postings_list)
            weight_term = log_N - log(docu_freq)

            for (doc_id, weight_docu) in postings_list:
                # compute tf.idf for term in document
                # would idf for queries just involve multiplying this by weight term again? since idf is same, tf should be 1 in query since we remove duplicates 
                scores[doc_id] += weight_term * weight_docu

        if len(scores) == 0:
            return ""

        print(dict(sorted(scores.items(), key=lambda item: item[1])))
        # exit()
        score_threshold = 4
        ids_to_return = [str(item[0]) for item in scores.items() if item[1]>score_threshold]

        # highest_scores = heapq.nlargest(number_results, scores.items(), key=lambda item: item[1])

        # # order by scores, then by term

        # highest_terms = []
        # same_scores = []
        # for item in highest_scores:
        #     # if score is the same as the previous highest score, add to list
        #     if highest_terms and item[1] == highest_terms[-1][1]:
        #         same_scores.append(item)
        #     # if score is different, sort the list of same scores by term
        #     elif same_scores:
        #         same_scores.sort(key=lambda x: x[0])
        #         highest_terms.extend(same_scores)
        #     else:
        #         highest_terms.append(item)
        
        # highest_terms = [str(item[0]) for item in highest_terms]

        return " ".join(ids_to_return)
    

    def fetch_postings_list(self, term):
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
            
            postings_list = [(tuple[0], tuple[1] * weight_term) for tuple in postings_list]
  
        return postings_list
    
    def process_query_boolean(self, query):
        bigrams = []
        #try: 
        tokens = word_tokenize(query)

        #remove punctuation 
        tokens = [term for term in tokens if term not in punctuation]

        #remove quotation marks - double check why theres 2 unique quotation marks strings
        tokens = [term for term in tokens if term != "``"]
        tokens = [term for term in tokens if term != "''"]
        tokens = [self.OPERATOR_AND if term == "AND" else term for term in tokens]

        #stem
        tokens = [self.stemmer.stem(term.lower()) if term != self.OPERATOR_AND else term for term in tokens]

        # convert terms to bigrams 
        # bigram logic currently evaluates "phone AND 'high court date'" into phone AND high court AND date

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
                if i >= len(items)-1:
                    final_items.append(self.OPERATOR_AND)
            elif len(item) == 2:
                final_items.append((item[0], item[1]))
                final_items.append(self.OPERATOR_OR)
                final_items.append(item[0])
                final_items.append(self.OPERATOR_OR)
                final_items.append(item[1])
                if i >= len(items)-1:
                    final_items.append(self.OPERATOR_AND)
            elif len(item) == 1:
                final_items.append(item[0])
                if i >= len(items)-1:
                    final_items.append(self.OPERATOR_AND)


                

        # # boolean needs to be at least 3 tokens i.e term AND term
        # if (len(tokens) > 2):
        #     #loop jumps through tokens with a skip of 2
        #     for i in range(0, len(tokens) - 1, 2):
        #         # case if 2 tokens found are (term, AND)
        #         if (tokens[i + 1] == self.OPERATOR_AND):
        #             #only append term if term in dictionary to avoid KeyError when evaluating
        #             if (tokens[i] in self.dictionary):
        #                 bigrams.append(tokens[i])
        #                 bigrams.append(self.OPERATOR_AND)
        #         # case if 2 tokens found are (AND, term) 
        #         elif (tokens[i] == self.OPERATOR_AND):
        #             if (tokens[i + 1] in self.dictionary):
        #                 bigrams.append(self.OPERATOR_AND)
        #                 bigrams.append(tokens[i + 1])
        #         # case if 2 tokens found are (term1, term 2)
        #         else:
        #             bigram = (tokens[i], tokens[i + 1])
        #             # only append bigram if in dictionary
        #             if bigram in self.dictionary:
        #                 # check for out of bounds error
        #                 # case if token before bigram (term1, term2) is not AND i.e. token was intially [term0, term1, term2] trigram
        #                 if (i != 0 and tokens[i - 1] != self.OPERATOR_AND):
        #                     # transforms [term0, term1, term2] to term0 AND (term1, term2) 
        #                     bigrams.append(self.OPERATOR_AND)
        #                     bigrams.append(bigram)
        #                 # check for out of bounds error
        #                 # case if token after bigram (term1, term2) is not AND i.e. token list was intially [term1, term2, term3] trigram
        #                 elif ((i + 1) != len(tokens) - 1 and tokens[i + 2] != self.OPERATOR_AND):
        #                     # transforms [term1, term2, term3] to (term1, term2) AND term3
        #                     bigrams.append(bigram)
        #                     bigrams.append(self.OPERATOR_AND)
        #                 else:
        #                     bigrams.append(bigram)     
        #     # if there are odd number of tokens, last token wont be covered by the initial for loop due to skip of 2
        #     # check if last token is in dictionary        
        #     if (len(tokens)%2 != 0) and tokens[len(tokens) - 1] in self.dictionary:
        #         # check that second last token wasnt AND e.g. initial token list was [term1, AND, term2]
        #         # append AND if above wasnt the case e.g. initial token list of [term1, AND, term2, term3, term 4] becomes term1 AND (term2, term3) AND term4
        #         if (tokens[len(tokens) - 2] != self.OPERATOR_AND):
        #             bigrams.append(self.OPERATOR_AND)
        #         # for the former case e.g. initial token list [term1, AND, term2] becomes term1 AND term 2
        #         bigrams.append(tokens[len(tokens) - 1])
        #     # set tokens list to bigrammed tokens list
        #     tokens = bigrams

        # put this here as token list might become less than 3 tokens after bigraming e.g. (term1, term2) AND 
        if len(final_items) < 3: 
            return ""

        print('final_items', final_items)
        
        postfix = self.convert_to_postfix(final_items)

        result = self.evaluate_postfix(postfix)

        # sort tuples of (docID, tf-idf) by tf-idf
        result = sorted(result, key = lambda x: x[1], reverse = True)
        print(result)

        #only return docIDs
        result = [str(r[0]) for r in result]   
        #except Exception as e:
            #return "ERROR" + str(e)
        
        return result


    # use shunting-yard algorithm to process query into postfix notation
    def convert_to_postfix(self, tokens):
        output_queue = []
        operator_stack = []

        for token in tokens:
            if token in self.OPERATOR_LIST:
                while (operator_stack and operator_stack[-1] >= token): # OR AND NOT will never be greater than parenthesis, omit check for parenthesis
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            else: # token is a term
                output_queue.append(token)

        while operator_stack:
            output_queue.append(operator_stack.pop())

        print(output_queue)

        return output_queue
    
    
    def evaluate_postfix(self, postfix):
        eval_stack = []
        for token in postfix:
            if token in self.OPERATOR_LIST:
                if token == self.OPERATOR_OR:
                    eval_stack.append(self.or_operation(eval_stack.pop(), eval_stack.pop()))
                elif token == self.OPERATOR_AND:
                    eval_stack.append(self.and_operation(eval_stack.pop(), eval_stack.pop()))
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
        log_N = log(self.dictionary[TOTAL_DOCUMENTS_KEY])

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
        
        return results_list


'''
if __name__ == "__main__":
    qp = QueryProcessor("data/struct_compress_dictionary", "data/struct_compress_postings")
    query = 'quiet phone call'

    print(qp.process_query(query))
'''