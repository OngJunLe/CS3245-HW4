import pickle
import heapq
import re
from collections import defaultdict
from math import log
from index import DOCUMENT_LENGTH_KEY, TOTAL_DOCUMENTS_KEY
from nltk import PorterStemmer, word_tokenize
from string import punctuation
from nltk.corpus import wordnet as wn

class QueryProcessor:
    OPERATOR_AND = 2
    regex_pattern = r'\bAND\b|\bOR\b|\bNOT\b|[\(\)]|[^\s()]+'
    OPERATOR_LIST = [OPERATOR_AND]

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
        bigrams = []
        query_terms = []
        scores = defaultdict(float)
        

        # tokenize and remove punctuation
        initial_terms = word_tokenize(query)
        
        initial_terms = [term for term in initial_terms if term not in punctuation]

        '''
        # remove duplicate terms and stem the terms - cant use set() since need to preserve order
        # dont need to remove though i think?
        for term in initial_terms:
            if term not in unique_terms:
                query_terms.append(term)
                unique_terms.append(term)
            
        
        # add relevant query terms with Wordnet
        for query in query_terms:
            query = wn.synsets(query)[0]
            for lemma in query.lemmas():
                to_add.append(lemma.name())
        query_terms = query_terms.union(set(to_add))
        '''
        
        # stem the terms
        query_terms = [self.stemmer.stem(term.lower()) for term in query_terms]

        
        # convert terms to bigrams
        if (len(query_terms) != 0):
            for i in range(len(query_terms) - 1):
                bigram = (query_terms[i], query_terms[i + 1])
                bigrams.append(bigram)
            query_terms = bigrams
        
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
                # weight_docu = (1 + log(term_freq))
                scores[doc_id] += weight_term * weight_docu

        '''
        #Removed normalization for now 
        for doc_id in scores.keys():
            scores[doc_id] /= self.document_length_dictionary[doc_id]
        '''
        if len(scores) == 0:
            return ""

        highest_scores = heapq.nlargest(number_results, scores.items(), key=lambda item: item[1])

        # order by scores, then by term

        highest_terms = []
        same_scores = []
        for item in highest_scores:
            # if score is the same as the previous highest score, add to list
            if highest_terms and item[1] == highest_terms[-1][1]:
                same_scores.append(item)
            # if score is different, sort the list of same scores by term
            elif same_scores:
                same_scores.sort(key=lambda x: x[0])
                highest_terms.extend(same_scores)
            else:
                highest_terms.append(item)
        
        highest_terms = [str(item[0]) for item in highest_terms]

        return " ".join(highest_terms)
    

    def fetch_postings_list(self, term):
        offset, bytes_to_read = self.dictionary[term]

        # read the postings list from the postings file
        with open(self.postings_file, 'rb') as f:
            f.seek(offset)
            postings_list = pickle.loads(f.read(bytes_to_read))

        return postings_list
    
    def process_query_boolean(self, query):
        bigrams = []
        try:
            # check if this lines up with nltk word_tokenize 
            tokens = list(self.tokenize_query(query))

            # check for any invalid tokens
            invalid_tokens = []
            for t in tokens:
                if t in self.dictionary:
                    if "&" in t or "|" in t or "~" in t:
                        invalid_tokens.append(t)
                else:
                    if t not in self.OPERATOR_LIST:
                        invalid_tokens.append(t)

            if len(invalid_tokens) > 0:
                return "invalid token(s): " + ", ".join(invalid_tokens)
            '''
            # convert terms to bigrams - need to account for operators 
            # double check logic for queries with 1-2 words
            if (len(tokens) != 0):
                for i in range(len(tokens) - 1):
                    if (tokens[i + 1] == self.OPERATOR_AND):
                        bigrams.append(self.OPERATOR_AND)
                    elif (tokens[i] == self.OPERATOR_AND):
                        continue
                    else:
                        bigram = (tokens[i], tokens[i + 1])
                        bigrams.append(bigram)
                tokens = bigrams
            
            #tokens = self.optimise_query(tokens) -- add back evaluating shorter postings lists first
            '''
            if len(tokens) == 0: return ""

            postfix = self.convert_to_postfix(tokens)

            # add sorting results by idf 
            result = self.evaluate_postfix(postfix)
            
        except Exception as e:
            return "ERROR" + str(e)
        
        return str(result)
        
        
    def tokenize_query(self, query):
        for token in re.findall(self.regex_pattern, query):
            if token == "AND":
                yield self.OPERATOR_AND
            else:
                yield self.stemmer.stem(token).lower()

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

        return output_queue
    
    
    def evaluate_postfix(self, postfix):
        eval_stack = []
        for token in postfix:
            if token in self.OPERATOR_LIST:
                eval_stack.append(self.and_operation(eval_stack.pop(), eval_stack.pop()))
            else:
                eval_stack.append(self.fetch_postings_list(token))
        return eval_stack[0]

    #need to add returning results by idf
    def and_operation(self, postings1, postings2):
        #node1 = postings1.head
        #node2 = postings2.head
        current_index_1 = 0
        current_index_2 = 0
        skip_1 = int(len(postings1) ** 0.5)
        skip_2 = int(len(postings2) ** 0.5)
        results_list = []
        # to keep track of the previous intersecting node in postings1
        # because postings1 will be modified in place
        #prev1 = None
        '''
        while node1 is not None and node2 is not None:
            if node1.data == node2.data:
                node1.skip = None # remove skip pointers
                node2 = node2.next
                prev1 = node1
                node1 = node1.next
            elif node1.data < node2.data:
                if node1.skip and node1.skip.data <= node2.data:
                    while node1.skip and node1.skip.data <= node2.data:
                        node1 = node1.skip
                # first intersection node should be made the new head
                if prev1 is None:
                    postings1.head = node1.next # remove node1 from postings1
                else:
                    prev1.next = node1.next # remove node1 from postings1
                node1 = node1.next
            else:
                if node2.skip and node2.skip.data <= node1.data:
                    while node2.skip and node2.skip.data <= node1.data:
                        node2 = node2.skip
                node2 = node2.next
            '''
        while current_index_1 < len(postings1) and current_index_2 < len(postings2):
            if (postings1[current_index_1] == postings2[current_index_2]):
                results_list.append(postings1[current_index_1][0])
                current_index_1 += 1
                current_index_2 += 1
            elif (postings1[current_index_1][0] < postings2[current_index_2][0]):
                if (current_index_1 + skip_1 < len(postings1)):
                    if (postings1[current_index_1 + skip_1][0] <= postings2[current_index_2][0]):
                        current_index_1 += skip_1
                    else:
                        current_index_1 += 1
                else:
                    current_index_1 += 1
            else:
                if (current_index_2 + skip_2 < len(postings2)):
                    if (postings2[current_index_2 + skip_2][0] <= postings1[current_index_1][0]):
                        current_index_2 += skip_2
                    else:
                       current_index_2 += 1
                else:
                    current_index_2 += 1
        return results_list

'''
qp = QueryProcessor("dictionary.txt", "postings.txt")
query = "phone AND call"
test1 = [(30, "thirty"), (40, "fourty"), (50, "fifty")]
test2 = [(20, "twenty"), (40, "fourty"), (60, "sixty")]
print(qp.process_query_boolean(query))
'''