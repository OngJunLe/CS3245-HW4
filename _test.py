import csv
import numpy as np
import sys
from nltk.tokenize import word_tokenize
from collections import Counter
from collections import defaultdict
import nltk
import string

def read_csv(file_path, no_rows=10):
    lines = []
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)

        i = 0
        for row in reader:
            if i >= no_rows:
                break
            else: i+=1
            lines.append(row)

    with open('data/sample.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(lines)


def count_csv(file_path):
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        row_count = sum(1 for row in reader)

        print(row_count)

def count_words(file_path):
    print('reading dataset...')

    # read csv file
    with open(file_path, 'r') as file:
        reader = csv.reader(file)

        term_index = defaultdict(lambda: len(term_index))
        # stopwords = set(nltk.corpus.stopwords.words('english'))
        stemmer = nltk.stem.PorterStemmer()
        # unique_term_ids = set()


        for line in reader:
            doc_id = line[0]
            content = line[2]
            
            # Tokenization: Convert text into tokens (words)
            words = word_tokenize(content)

            words = [stemmer.stem(word).lower() for word in words if word not in string.punctuation] 

            tokens_id = [term_index[word] for word in words]
        
    # print keys of term_index into a .txt file
    with open('data/term_index.txt', 'w') as file:
        for key in term_index.keys():
            file.write(f"{key}\n")


            


def Estepvcompare():
    

    num_documents=10
    num_terms=20
    num_topics=2

    term_given_topic = np.random.rand(num_topics, num_terms)
    topic_given_doc = np.random.rand(num_documents, num_topics)
    P_z_dw = np.zeros((num_documents, num_terms, num_topics))
    P_z_dw2 = np.zeros((num_documents, num_terms, num_topics))
    
    # E-step: Calculate P(z|d,w)  
    for i in range(0, num_documents):
        for j in range(0, num_terms):
            denominator = 0
            for k in range(0, num_topics):
                P_z_dw[i, j, k] = term_given_topic[k, j] * topic_given_doc[i, k]
                denominator += P_z_dw[i, j, k]
            if denominator == 0:
                for k in range(0, num_topics):
                    P_z_dw[i, j, k] = 0
            else:
                for k in range(0, num_topics):
                    P_z_dw[i, j, k] /= denominator

    # print(P_z_dw[0,0,0])
    topic_given_doc_ext = topic_given_doc[:, np.newaxis, :]

    # Reshape and extend term_given_topic to (1, num_terms, num_topics) and then transpose to (1, num_topics, num_terms)
    term_given_topic_ext = term_given_topic.T[np.newaxis, :, :]

    # Element-wise multiplication across the extended dimensions
    P_z_dw2 = term_given_topic_ext * topic_given_doc_ext

    # Sum across the last dimension (num_topics) to get the denominator for each document and term
    denominator = np.sum(P_z_dw, axis=2, keepdims=True)

    # Use the denominator to normalize P_z_dw
    # Avoid division by zero by using np.where
    P_z_dw2 = np.where(denominator != 0, P_z_dw / denominator, 0)

    # P_z_dw2 = np.where(denominator == 0, 0, P_z_dw2)

    for i in range(0, num_documents):
        for j in range(0, num_terms):
            for k in range(0, num_topics):
                # print(f"{i}, {j}, {k}")
                # print(P_z_dw[i, j, k])
                # print(P_z_dw2[i, j, k])
                # check if P_z_dw and P_z_dw2 are approximately equal
                assert np.isclose(P_z_dw[i, j, k], P_z_dw2[i, j, k]), f"Error: {P_z_dw[i,j,k]} != {P_z_dw2[i,j,k]}"

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

def test_pairwise():
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # Assuming your matrix is called 'matrix' and the vector is called 'vector'
    # matrix is n by m, vector is length m

    # Generate some sample data
    n, m = 6, 4
    matrix = np.random.rand(n, m)
    vector = np.random.rand(m)

    # Calculate cosine similarity between the vector and each row of the matrix
    # Reshape the vector to match the matrix's dimensions
    vector_reshaped = vector.reshape(1, -1)  # Reshape to 1xM
    similarities = cosine_similarity(matrix, vector_reshaped)

    print(matrix)
    print(vector)

    # Print the cosine similarities
    print(similarities)

    threshold = 0.5
    for s in similarities:
        if s>threshold:
            print(s)


if __name__ == "__main__":
    # read_csv('data/dataset.csv', 100)
    # set_max_size()
    # count_words('data/dataset.csv')
    test_pairwise()

    
                
