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
import scipy.sparse
import numpy as np

from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from collections import Counter
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Format of term dictionary {term: (offset, no_bytes), ...}
# Format of postings [(docID, log term frequency), ...]

# Unique keys for document length dictionary and total number of documents
DOCUMENT_LENGTH_KEY = -100
TOTAL_DOCUMENTS_KEY = -200

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d temp_postings-file -p postings-file")


def build_index():
    print('reading dataset...')

    # read csv file
    with open('data/dataset.csv', 'r') as file:
        reader = csv.reader(file)
        # data = list(reader)

        max_lines = 18000
        i=0

        term_index = defaultdict(lambda: len(term_index))
        document_index = defaultdict(lambda: len(document_index))
        vocabulary = {}
        document_ref = {}
        stopwords = set(nltk.corpus.stopwords.words('english'))
        stemmer = nltk.stem.PorterStemmer()
        
        # store rows, columns and data for scipy sparse matrix
        rows = []
        cols = []
        data = []

        next(reader)  # exclude first line of column headers from indexing

        for line in reader:
            i+=1
            # print(i)
            if i>max_lines:
                break
            doc_id = line[0]
            content = line[2]
            
            # Tokenization: Convert text into tokens (words)
            tokens = word_tokenize(content)

            # TEMP - Remove any token with punctuation
            tokens = [word for word in tokens if not any(char in string.punctuation for char in word)]

            # Stop Words Removal: Remove common words that do not contribute to the meaning of the text.
            tokens = [word for word in tokens if word.lower() not in stopwords]

            # Stemming/Lemmatization: Reduce words to their base or root form.
            tokens = [stemmer.stem(word) for word in tokens]

            # Store the tf and row and column indices for the sparse matrix
            word_count = Counter(tokens)
            for word, count in word_count.items():
                data.append(1 + math.log10(count))
                rows.append(document_index[doc_id])
                cols.append(term_index[word])
                vocabulary[term_index[word]] = word
                document_ref[document_index[doc_id]] = doc_id

    print('constructing sparse matrix')

    # Create the sparse matrix
    num_documents = len(document_index)
    num_terms = len(term_index)

    # sparsity with first 1000 documents = 98.56%
    document_term_matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_documents, num_terms))

    doc_term_matrix_csr = document_term_matrix.tocsr()
    doc_term_matrix_csc = document_term_matrix.tocsc()

    # initialise the matrices for the pLSA algorithm
    num_topics = 10
    topic_given_doc = np.random.rand(num_documents,num_topics)
    topic_given_doc = topic_given_doc / topic_given_doc.sum(axis=1)[:, np.newaxis]

    term_given_topic = np.random.rand(num_topics,num_terms)
    term_given_topic = term_given_topic / term_given_topic.sum(axis=1)[:, np.newaxis]

    # P_z_dw = np.zeros((num_documents, num_terms, num_topics))

    # parameters for  pLSA algorithm
    max_iterations = 100
    tolerance = 1e-4
    previous_log_likelihood = -np.inf

    print('iterating...')

    progress_interval = max_iterations // 10
    progress_points = range(progress_interval, max_iterations, progress_interval)

    # EM Algorithm
    for iteration in range(max_iterations):
        # print progress every 10% of max_iterations:
        if iteration in progress_points:
            print(f"{iteration} iterations completed. Current log-likelihood: {log_likelihood}")

        # E-step: Calculate P(z|d,w)  
        # for i in range(0, num_documents):
        #     for j in range(0, num_terms):
        #         denominator = 0
        #         for k in range(0, num_topics):
        #             P_z_dw[i, j, k] = term_given_topic[k, j] * topic_given_doc[i, k]
        #             denominator += P_z_dw[i, j, k]
        #         if denominator == 0:
        #             for k in range(0, num_topics):
        #                 P_z_dw[i, j, k] = 0
        #         else:
        #             for k in range(0, num_topics):
        #                 P_z_dw[i, j, k] /= denominator

        # E-step v2: Calculate P(z|d,w) using vectorized operations
        # Calculate P(z|d,w) using vectorized operations
        # P_z_dw initially filled with term_given_topic * topic_given_doc for each (d, w)
        # term_given_topic_ext = term_given_topic[np.newaxis, :, :]

        # Extend topic_given_doc to span across all terms
        # topic_given_doc is originally (num_documents, num_topics)
        # We want it to be (num_documents, num_terms, num_topics) for broadcasting
        topic_given_doc_ext = topic_given_doc[:, np.newaxis, :]

        # Reshape and extend term_given_topic to (1, num_terms, num_topics) and then transpose to (1, num_topics, num_terms)
        term_given_topic_ext = term_given_topic.T[np.newaxis, :, :]

        # Element-wise multiplication across the extended dimensions
        P_z_dw = term_given_topic_ext * topic_given_doc_ext

        # Sum across the last dimension (num_topics) to get the denominator for each document and term
        denominator = np.sum(P_z_dw, axis=2, keepdims=True)

        denominator_nans = np.sum(np.isnan(denominator))
        denominator_zeros = np.sum(denominator==0)
        if denominator_nans or denominator_zeros:
            print(f"denominator_nans: {denominator_nans}, denominator_zeros: {denominator_zeros}")

        # Use the denominator to normalize P_z_dw
        # Avoid division by zero by using np.where
        P_z_dw = np.where(denominator != 0, P_z_dw / denominator, 0)


        # M-step: Update P(w|z) and P(d|z)
        # update term_given_topic
        for k in range(0, num_topics):
            denominator = 0
            for j in range(0, num_terms):
                term_given_topic[k, j] = 0
                # for the current term, retrieve the term frequency for each document
                indices = doc_term_matrix_csc.indices[doc_term_matrix_csc.indptr[j]:doc_term_matrix_csc.indptr[j+1]]
                counts = doc_term_matrix_csc.data[doc_term_matrix_csc.indptr[j]:doc_term_matrix_csc.indptr[j+1]]

                for i, d in enumerate(indices):
                    term_given_topic[k, j] += counts[i] * P_z_dw[d, j, k]
                denominator += term_given_topic[k, j]

            if denominator == 0:
                for j in range(0, num_terms):
                    term_given_topic[k, j] = 1.0 / num_terms
            else:
                for j in range(0, num_terms):
                    term_given_topic[k, j] /= denominator
        
        # update topic_given_doc
        for i in range(0, num_documents):
            for k in range(0, num_topics):
                topic_given_doc[i, k] = 0
                denominator = 0
                # for the current document, retrieve the term frequency for each term
                indices = doc_term_matrix_csr.indices[doc_term_matrix_csr.indptr[i]:doc_term_matrix_csr.indptr[i+1]]
                counts = doc_term_matrix_csr.data[doc_term_matrix_csr.indptr[i]:doc_term_matrix_csr.indptr[i+1]]

                for j, w in enumerate(indices):
                    topic_given_doc[i, k] += counts[j] * P_z_dw[i, w, k]
                    denominator += counts[j]

                if denominator == 0:
                    topic_given_doc[i, k] = 1.0 / num_topics
                else:
                    topic_given_doc[i, k] /= denominator

        # Normalize the matrices
        # topic_given_doc = topic_given_doc / topic_given_doc.sum(axis=1)[:, np.newaxis]
        # term_given_topic = term_given_topic / term_given_topic.sum(axis=0)

        # Check for convergence: Calculate log likelihood
        log_likelihood = 0
        for d in range(num_documents):
            indices = doc_term_matrix_csr.indices[doc_term_matrix_csr.indptr[d]:doc_term_matrix_csr.indptr[d+1]]
            counts = doc_term_matrix_csr.data[doc_term_matrix_csr.indptr[d]:doc_term_matrix_csr.indptr[d+1]]
            for i, w in enumerate(indices):
                p_w_d = np.sum(topic_given_doc[d, :] * term_given_topic[:, w])
                log_likelihood += counts[i] * np.log(p_w_d)
        

        if np.abs(log_likelihood - previous_log_likelihood) < tolerance:
            print(f"Convergence reached after {iteration+1} iterations.")
            break

        previous_log_likelihood = log_likelihood

    print("Final log likelihood:", log_likelihood)

    # test the model
    test_query = "good grades exchange scandal"

    query_tokens = tokenize(test_query, stemmer, stopwords)
    query_vector = np.zeros(num_terms)
    query_count = Counter(query_tokens)
    for word, count in query_count.items():
        if word in term_index:
            query_vector[term_index[word]] += (1+math.log10(count))

    query_topics = term_given_topic @ query_vector

    # match the closest document to the query
    # for doc_topics in topic_given_doc:
    #     score = np.dot(doc_topics, query_topics)

    query_topics = query_topics.reshape(1, -1)
    similarities = cosine_similarity(topic_given_doc, query_topics)
    # print(len(similarities))
    
    import heapq
    # Assuming similarities is a 1D array containing similarity scores for each document
    top_n_heap = []
    heapq.heapify(top_n_heap)  # Creates an empty heap

    for doc_index, similarity in enumerate(similarities):
        doc_id = document_ref[doc_index]
        # Use a tuple (similarity, doc_index) to keep track of document indices
        if len(top_n_heap) < 10:
            heapq.heappush(top_n_heap, (similarity, doc_id))
        elif similarity > top_n_heap[0][0]:  # Only push to heap if better than the smallest
            heapq.heappushpop(top_n_heap, (similarity, doc_id))

    # The heap now contains the top 10 elements with the smallest item at the root
    # To get the list in descending order
    top_n_docs = sorted(top_n_heap, key=lambda x: -x[0])

    # Extract just the document indices if needed
    top_n_doc_ids = [(doc_id, similarity) for similarity, doc_id in top_n_docs]

    print(top_n_doc_ids)

            
    return

def tokenize(query, stemmer, stopwords):
    # Tokenization: Convert text into tokens (words)
    tokens = word_tokenize(query)

    # TEMP - Remove any token with punctuation
    tokens = [word for word in tokens if not any(char in string.punctuation for char in word)]

    # Stop Words Removal: Remove common words that do not contribute to the meaning of the text.
    tokens = [word for word in tokens if word.lower() not in stopwords]

    # Stemming/Lemmatization: Reduce words to their base or root form.
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

def e_step():
    return

def set_max_size():
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10 
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

if __name__ == "__main__":
    set_max_size()
    build_index()


# Vectorization: Convert text into a numerical format, typically using TF-IDF (Term Frequency-Inverse Document Frequency) which reflects how important a word is to a document in a collection.

 # E-step: Calculate P(z|d,w)
        # P_z_dw = np.zeros((num_documents, num_terms, num_topics))
        # for d in range(num_documents):
        #     indices = doc_term_matrix.indices[doc_term_matrix.indptr[d]:doc_term_matrix.indptr[d+1]]
        #     counts = doc_term_matrix.data[doc_term_matrix.indptr[d]:doc_term_matrix.indptr[d+1]]
        #     for i, w in enumerate(indices):
        #         prob = topic_given_doc[d, :] * term_given_topic[:, w]
        #         P_z_dw[d, w, :] = prob / prob.sum()
        #     print(P_z_dw[d, :, :])

# M-step: Update P(w|z) and P(z|d)
        # topic_given_doc.fill(0)
        # term_given_topic.fill(0)
        # for d in range(num_documents):
        #     indices = doc_term_matrix.indices[doc_term_matrix.indptr[d]:doc_term_matrix.indptr[d+1]]
        #     counts = doc_term_matrix.data[doc_term_matrix.indptr[d]:doc_term_matrix.indptr[d+1]]
        #     for i, w in enumerate(indices):
        #         topic_given_doc[d, :] += counts[i] * P_z_dw[d, w, :]
        #         term_given_topic[w, :] += counts[i] * P_z_dw[d, w, :]