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
# import dask.dataframe as dd
import dask.array as da
import dask.bag as db
from dask.delayed import delayed
import pandas as pd
import dask
from dask.distributed import Client

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
    print('initializing dask client')
    client = Client()
    print(client)

    print('reading dataset...')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stemmer = nltk.stem.PorterStemmer()

    df = pd.read_csv('data/dataset.csv', sep=',', header=0, quotechar='"', quoting=csv.QUOTE_ALL)

    # dataset = dd.from_pandas(df, chunksize=100)
    bag = db.from_sequence(df['content'][:100])
    token_counter_list = bag.map(tokenize, stemmer=stemmer, stopwords=stopwords).compute()

    doc_ids = list(df['document_id'][:100])

    # result = dataset.map_partitions(tokenize_partition, stemmer=stemmer, stopwords=stopwords).compute()
    # output_format = {'counter_tokens': Counter}
    # dataset['counter_tokens'] = dataset.apply(tokenize, stemmer=stemmer, stopwords=stopwords, meta=output_format, axis=1).compute()
    # dataset['counter_tokens'] = tokenize(dataset['content'], stemmer, stopwords)
    # result =result.compute()
    

    term_index = defaultdict(lambda: len(term_index))
    document_index = defaultdict(lambda: len(document_index))
    vocabulary = {}
    document_ref = {}
    
    # store rows, columns and data for scipy sparse matrix
    rows = []
    cols = []
    data = []

    for doc_id, word_count in zip(doc_ids, token_counter_list):
        # doc_id = row['document_id']
        
        # Stemming/Lemmatization: Reduce words to their base or root form.
        # word_count = row['counter_tokens']

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

    document_term_matrix = document_term_matrix.todense()
    document_term_array = np.asarray(document_term_matrix)
    nrow, ncol = document_term_matrix.shape

    # document_term_matrix_cols = da.from_array(document_term_matrix, chunks=(nrow,1))
    # document_term_matrix_rows = da.from_array(document_term_matrix, chunks=(ncol,1))

    # doc_term_matrix_csr = document_term_matrix.tocsr()
    # doc_term_matrix_csc = document_term_matrix.tocsc()

    # initialise the matrices for the pLSA algorithm
    num_topics = 10
    topic_given_doc = np.random.rand(num_documents,num_topics)
    topic_given_doc = da.from_array(topic_given_doc)
    topic_given_doc = topic_given_doc / topic_given_doc.sum(axis=1)[:, np.newaxis]

    term_given_topic = np.random.rand(num_topics,num_terms)
    term_given_topic = da.from_array(term_given_topic)
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
        print(iteration)
        # print progress every 10% of max_iterations:
        if iteration in progress_points:
            print(f"{iteration} iterations completed. Current log-likelihood: {log_likelihood}")

        # E-step v2: Calculate P(z|d,w) using vectorized operations
        # Calculate P(z|d,w) using vectorized operations
        # P_z_dw initially filled with term_given_topic * topic_given_doc for each (d, w)

        # Extend topic_given_doc to span across all terms
        # topic_given_doc is originally (num_documents, num_topics)
        # We want it to be (num_documents, num_terms, num_topics) for broadcasting
        topic_given_doc_ext = topic_given_doc[:, np.newaxis, :]

        # Reshape and extend term_given_topic to (1, num_terms, num_topics) and then transpose to (1, num_topics, num_terms)
        term_given_topic_ext = term_given_topic.T[np.newaxis, :, :]

        # Element-wise multiplication across the extended dimensions
        P_z_dw = term_given_topic_ext * topic_given_doc_ext

        # Sum across the last dimension (num_topics) to get the denominator for each document and term
        denominator = da.sum(P_z_dw, axis=2, keepdims=True)

        denominator_nans = np.sum(np.isnan(denominator))
        denominator_zeros = np.sum(denominator==0)
        if denominator_nans or denominator_zeros:
            print(f"denominator_nans: {denominator_nans}, denominator_zeros: {denominator_zeros}")

        # Use the denominator to normalize P_z_dw
        # Avoid division by zero by using np.where
        P_z_dw = np.where(denominator != 0, P_z_dw / denominator, 0)

        print('e step complete')

        # M-step: Update P(w|z) and P(d|z)
        # update term_given_topic
        def process_term_frequencies(j):
            values = document_term_array[:,j]
            values_ext = values[:,np.newaxis] # shape num_doc, num_topics
            result = values_ext * P_z_dw[:,j,:]
            result = result.sum(axis=0)

            return result # shape num_topics

        # assuming block is one column
        def process_term_frequencies_block(values):
            values_ext = values[np.newaxis, :, np.newaxis]
            result = values_ext * P_z_dw[:,j,:]
            return result

        results = [delayed(process_term_frequencies)(j) for j in range(num_terms)]
        results = dask.compute(*results)
        print('term frequencies processed')
        # results = document_term_matrix_cols.map_blocks(process_term_frequencies_block)
        # sum along the document axis
        
        stacked_results = da.stack(results,axis=1) # shape num_topics, num_terms
        # combined_results = da.sum(stacked_results, axis=1)
        denominators = stacked_results.sum(axis=0, keepdims=False)
        mask = denominators==0
        # print('checking shapes')
        # print(stacked_results.shape)
        # print(combined_results.shape)
        # print(denominators.shape)
        # print(mask.shape)
        denominators[mask] = 1
        normalized_results = stacked_results / denominators
        # mask = mask.flatten()
        normalized_results[:,mask] = 1/num_terms
        term_given_topic = normalized_results.compute()
        print('term_given_topic computed')
        
        # update topic_given_doc
        def process_topic_given_doc(i):
            values = document_term_array[i,:]
            values_ext = values[:, np.newaxis] # shape num_term, num_topic
            result = values_ext * P_z_dw[i,:,:] # shape num_term, num_topic
            print('values_ext', values_ext.shape)
            print('P_z_dw[i,:,:]', P_z_dw[i,:,:].shape)
            print('result', result.shape)
            exit()
            return result

        def process_topic_given_doc_block(values):
            values_ext = values[:,:, np.newaxis]
            result = values_ext * P_z_dw[i,:,:]
            return result

        # results = document_term_matrix_rows.map_blocks(process_topic_given_doc_block)
        results = [delayed(process_topic_given_doc)(i) for i in range(num_documents)]
        results = dask.compute(*results)
        # sum along the term axis
        combined_results = da.sum(da.stack(results), axis=1)
        denominators = combined_results.sum(axis=0, keepdims=False)
        mask = denominators==0
        denominators[mask] = 1
        normalized_results = combined_results / denominators
        # mask = mask.flatten()
        normalized_results[mask,:] = 1/num_documents
        topic_given_doc = normalized_results.compute()

        # Check for convergence: Calculate log likelihood
        log_likelihood = 0
        def count_logll(d):
            values = document_term_matrix[d,:] # shape (num_terms)
            p_w_d = topic_given_doc[d, :] @ term_given_topic # shape (1, num_topics) @ (num_topics, num_terms)
            return da.sum(p_w_d * values)
        
        results = [delayed(count_logll)(d) for d in range(num_documents)]
        results = dask.compute(*results)
        # results = document_term_matrix_rows.map_blocks(count_logll)
        log_likelihood = da.sum(da.stack(results))        

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

def tokenize_partition(df, stemmer, stopwords):
    # output_format = {'counter_tokens': Counter}
    # df['counter_tokens'] = df['content'].apply(meta=output_format, axis=1, func=lambda x: tokenize(x, stemmer, stopwords))
    # df['counter_tokens'] = df['content'].apply(tokenize, stemmer=stemmer, stopwords=stopwords, meta=output_format, axis=1)

    df['processed'] = df['contents'].apply(tokenize, stemmer=stemmer, stopwords=stopwords, meta=('contents', 'object'))
    return df

def tokenize(query, stemmer, stopwords):
    # print('query: ', query)
    # Tokenization: Convert text into tokens (words)
    tokens = word_tokenize(query)

    # TEMP - Remove any token with punctuation
    tokens = [word for word in tokens if not any(char in string.punctuation for char in word)]

    # Stop Words Removal: Remove common words that do not contribute to the meaning of the text.
    tokens = [word for word in tokens if word.lower() not in stopwords]

    # Stemming/Lemmatization: Reduce words to their base or root form.
    tokens = [stemmer.stem(word) for word in tokens]
    return Counter(tokens)

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