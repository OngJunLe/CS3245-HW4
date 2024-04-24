from numpy import zeros, int8, log, int16
from pylab import random
import sys
# import jieba
import csv
import re
import time
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import word_tokenize
import string
import scipy
# import numpy as np
# import codecs

### TAKEN FROM https://github.com/laserwave/plsa/

# segmentation, stopwords filtering and document-word matrix generating
# [return]:
# N : number of documents
# M : length of dictionary
# word2id : a map mapping terms to their corresponding ids
# id2word : a map mapping ids to terms
# X : document-word matrix, N*M, each line is the number of terms that show up in the document
def preprocessing(datasetFilePath, stopwordsFilePath):
    
    # read the stopwords file
    # file = codecs.open(stopwordsFilePath, 'r', 'utf-8')
    # stopwords = [line.strip() for line in file] 
    # file.close()
    
    # # read the documents
    # file = codecs.open(datasetFilePath, 'r', 'utf-8')
    # documents = [document.strip() for document in file] 
    # file.close()

    file_path = 'data/dataset.csv'

    with open(file_path, 'r') as file:
        reader = csv.reader(file)

        word2id = defaultdict(lambda: len(word2id))
        id2word = {}
        id2index = defaultdict(lambda: len(id2index))
        # stopwords = set(nltk.corpus.stopwords.words('english'))
        stemmer = nltk.stem.PorterStemmer()
        # unique_term_ids = set()
        i = 0
        max_rows = 100
        # store rows, columns and data for scipy sparse matrix
        rows = []
        cols = []
        data = []

        # X = zeros([max_rows, max_rows], int16)

        for line in reader:
            i+=1
            if i>max_rows:
                break
            doc_id = line[0]
            content = line[2]
            
            # Tokenization: Convert text into tokens (words)
            words = word_tokenize(content)

            tokens = [stemmer.stem(word).lower() for word in words if word not in string.punctuation] 

            word_count = Counter(tokens)
            for word, count in word_count.items():
                data.append(1 + log(count))
                rows.append(id2index[doc_id])
                cols.append(word2id[word])
                id2word[word2id[word]] = word


    # number of documents
    # N = len(documents)

    # Create the sparse matrix
    num_documents = len(id2index)
    num_terms = len(word2id)

    # sparsity with first 1000 documents = 98.56%
    document_term_matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_documents, num_terms))
    X = document_term_matrix.toarray()

    # doc_term_matrix_csr = document_term_matrix.tocsr()
    # doc_term_matrix_csc = document_term_matrix.tocsc()

    # length of dictionary
    # M = len(word2id)  

    # generate the document-word matrix
    # X = zeros([N, M], int8)
    # for word in word2id.keys():
    #     j = word2id[word]
    #     for i in range(0, N):
    #         if word in wordCounts[i]:
    #             X[i, j] = wordCounts[i][word]    

    return num_documents, num_terms, word2id, id2word, X

def initializeParameters():
    for i in range(0, N):
        normalization = sum(lamda[i, :])
        for j in range(0, K):
            lamda[i, j] /= normalization

    for i in range(0, K):
        normalization = sum(theta[i, :])
        for j in range(0, M):
            theta[i, j] /= normalization

def EStep():
    for i in range(0, N):
        for j in range(0, M):
            denominator = 0
            for k in range(0, K):
                p[i, j, k] = theta[k, j] * lamda[i, k]
                denominator += p[i, j, k]
            if denominator == 0:
                for k in range(0, K):
                    p[i, j, k] = 0
            else:
                for k in range(0, K):
                    p[i, j, k] /= denominator

def MStep():
    # update theta
    for k in range(0, K):
        denominator = 0
        for j in range(0, M):
            theta[k, j] = 0
            for i in range(0, N):
                theta[k, j] += X[i, j] * p[i, j, k]
            denominator += theta[k, j]
        if denominator == 0:
            for j in range(0, M):
                theta[k, j] = 1.0 / M
        else:
            for j in range(0, M):
                theta[k, j] /= denominator
        
    # update lamda
    for i in range(0, N):
        for k in range(0, K):
            lamda[i, k] = 0
            denominator = 0
            for j in range(0, M):
                lamda[i, k] += X[i, j] * p[i, j, k]
                denominator += X[i, j]
            if denominator == 0:
                lamda[i, k] = 1.0 / K
            else:
                lamda[i, k] /= denominator

# calculate the log likelihood
def LogLikelihood():
    loglikelihood = 0
    for i in range(0, N):
        for j in range(0, M):
            tmp = 0
            for k in range(0, K):
                tmp += theta[k, j] * lamda[i, k]
            if tmp > 0:
                loglikelihood += X[i, j] * log(tmp)
    return loglikelihood

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

# output the params of model and top words of topics to files
# def output():
#     # document-topic distribution
#     file = codecs.open(docTopicDist,'w','utf-8')
#     for i in range(0, N):
#         tmp = ''
#         for j in range(0, K):
#             tmp += str(lamda[i, j]) + ' '
#         file.write(tmp + '\n')
#     file.close()
    
#     # topic-word distribution
#     file = codecs.open(topicWordDist,'w','utf-8')
#     for i in range(0, K):
#         tmp = ''
#         for j in range(0, M):
#             tmp += str(theta[i, j]) + ' '
#         file.write(tmp + '\n')
#     file.close()
    
#     # dictionary
#     file = codecs.open(dictionary,'w','utf-8')
#     for i in range(0, M):
#         file.write(id2word[i] + '\n')
#     file.close()
    
#     # top words of each topic
#     file = codecs.open(topicWords,'w','utf-8')
#     for i in range(0, K):
#         topicword = []
#         ids = theta[i, :].argsort()
#         for j in ids:
#             topicword.insert(0, id2word[j])
#         tmp = ''
#         for word in topicword[0:min(topicWordsNum, len(topicword))]:
#             tmp += word + ' '
#         file.write(tmp + '\n')
#     file.close()

set_max_size()    

# set the default params and read the params from cmd
datasetFilePath = 'dataset.txt'
stopwordsFilePath = 'stopwords.dic'
K = 10    # number of topic
maxIteration = 30
threshold = 10.0
topicWordsNum = 10
docTopicDist = 'docTopicDistribution.txt'
topicWordDist = 'topicWordDistribution.txt'
dictionary = 'dictionary.dic'
topicWords = 'topics.txt'
if(len(sys.argv) == 11):
    datasetFilePath = sys.argv[1]
    stopwordsFilePath = sys.argv[2]
    K = int(sys.argv[3])
    maxIteration = int(sys.argv[4])
    threshold = float(sys.argv[5])
    topicWordsNum = int(sys.argv[6])
    docTopicDist = sys.argv[7]
    topicWordDist = sys.argv[8]
    dictionary = sys.argv[9]
    topicWords = sys.argv[10]

# preprocessing
N, M, word2id, id2word, X = preprocessing(datasetFilePath, stopwordsFilePath)

# lamda[i, j] : p(zj|di)
lamda = random([N, K])

# theta[i, j] : p(wj|zi)
theta = random([K, M])

# p[i, j, k] : p(zk|di,wj)
p = zeros([N, M, K])

initializeParameters()

# EM algorithm
oldLoglikelihood = 1
newLoglikelihood = 1
for i in range(0, maxIteration):
    EStep()
    MStep()
    newLoglikelihood = LogLikelihood()
    print(newLoglikelihood)
    print("[", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "] ", i+1, " iteration  ", str(newLoglikelihood))
    if(oldLoglikelihood != 1 and newLoglikelihood - oldLoglikelihood < threshold):
        break
    oldLoglikelihood = newLoglikelihood

# output()