import pickle
import heapq
from collections import defaultdict
from math import log
from index import DOCUMENT_LENGTH_KEY, TOTAL_DOCUMENTS_KEY
from nltk import PorterStemmer, word_tokenize
from string import punctuation
from nltk.corpus import wordnet as wn

class QueryProcessor:

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
        unique_terms = []
        scores = defaultdict(float)
        

        # tokenize and remove punctuation
        initial_terms = word_tokenize(query)
        
        initial_terms = [term for term in initial_terms if term not in punctuation]

        
        # remove duplicate terms and stem the terms - cant use set() since need to preserve order
        for term in initial_terms:
            if term not in unique_terms:
                query_terms.append(term)
                unique_terms.append(term)
            
        '''
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
    


