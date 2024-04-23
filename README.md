This is the README file for A0216276A's, A0291640H's submission
Email(s): e0538377@u.nus.edu, e1332814@u.nus.edu

== Python Version ==

We're using Python Version 3.11.2 for this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

CSV file was processed with Pandas and Dask to extract docIDs and content.

build_index() in index.py hashes terms and postings lists of [(docID, log term frequency), ...] into the temporary postings dictionary in memory, which is then written to disk as postings.txt once all files in the input directory have been processed. For this assignments, terms include both single, as well as bigram terms, to account for phrase queries in boolean search. burton_thesaurus.txt is also processed in indexing, to generate a dictionary with keys of legal terms and values of synonomous terms, to be used in search for query refinement. 3 files are outputted: dictionary.txt, with a dictionary of (term, (byte offset of posting list in postings.txt, number of bytes of postings list)), postings.txt, as previously mentioned, and binary_thesaurus.txt, all serialised with Pickle. The byte offset approach ensures that postings lists can be read in search.py without reading the entire postings.txt. A variable for the total number of documents is also added into dictionary.txt with unique keys to allow for IDF calculation in the search process.

For search:
QueryProcessor class is defined in query_processor.py, handles an individual query using the process_query() or process_query_boolean() method. 

process_query() works as follows as follows: word_tokenize() -> remove punctuation -> remove duplicate terms -> lower() -> stem()
invalid terms (i.e. not defined in dictionary in indexing) are removed. postings list for each term are retrieved, used to calculate tf-idf scores following algorithm in lecture notes. Cosine normalization was not employed, as due to the nature of legal documents, it is possible that important terms e.g. court name might only occur a few times in a lengthy document.
Scores are returned in sorted order, documents with equal scores are sorted by document number.

process_query_boolean() works as follows: split the string into a list of tokens of operators/terms, remove punctuation and stem, process phrases to bigrams where necessary, convert the order of tokens to be in postfix notation, evaluate the query, loading postings from disk as needed.
After evaluating, tf-idf scores were calculated as in free text search, and scores are returned in sorted order, documents with equal scores are sorted by document number. 

search.py handles the processing of the query and output file by calling either process_query or process_query_boolean depending on if "AND" exists in the query string, as well as sorting the results from query_processor to ensure that any relevant documents, as specified in the queries file, are returned first.

== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

index.py: Code for indexing logic.
search.py: Code for search logic. 
query_processor.py: Helper class for processing search. 
dictionary.txt: Dictionary serialised with Pickle. 
postings.txt: Posting lists serialised with Pickle.
burtons_thesaurus.txt: Txt file for legal thesaurus used for query refinement.
binary_thesaurus.txt: Binary file of dictionary generated from burtons_thesaurus.txt

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[ X ] We, A0216276A and A0291640H, certify that we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, we
expressly vow that we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I/We, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

We suggest that we should be graded as follows:

<Please fill in>

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>

Dask, Pandas, site for legal thesaurus and parsing code 
