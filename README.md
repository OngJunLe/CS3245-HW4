This is the README file for A0216276A's, A0291640H's submission
Email(s): e0538377@u.nus.edu, e1332814@u.nus.edu

== Python Version ==

We're using Python Version 3.11.2 for this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

CSV file was processed with Pandas to extract courts, docIDs and content, with Dask used to parallelize the process.

build_index() in index.py hashes terms and postings lists of [(docID, log term frequency), ...] into the temporary postings dictionary in memory, which is then written to disk as postings.txt once all files in the input directory have been processed. For this assignment, terms include both single, as well as bigram terms, to account for phrase queries in boolean search. Gurton_thesaurus.txt is also processed in indexing, to generate a dictionary with keys of legal terms and values of synonomous terms, to be used in search for query refinement. 3 files are outputted: dictionary.txt, with a dictionary of (term, (byte offset of posting list in postings.txt, number of bytes of postings list)), postings.txt, as previously mentioned, and binary_thesaurus.txt, all serialised with Pickle. The byte offset approach ensures that postings lists can be read in search.py without reading the entire postings.txt. A variable for the total number of documents is also added into dictionary.txt with unique keys to allow for IDF calculation in the search process.

For search:
QueryProcessor class is defined in query_processor.py, handles an individual query using the process_query() or process_query_boolean() method. 

For both query types, query expansion was implemented through the use of Burton's Legal Thesaurus, to expand queries with terms more specific to the legal field. The txt file of this thesaurus was processed into binary_thesaurus.txt in index.py with Pickle, with some lines in the original thesaurus such as "FOREIGN PHRASES" skipped for brevity, as well as terms with len > 3 skipped for ease of integrating new query terms as bigram phrase queries. Zones were also implemented to put a larger weight on courts and titles in documents.

process_query() works as follows as follows: word_tokenize() -> remove punctuation -> remove duplicate terms -> lower() -> stem() -> query expansion -> evaluate query.
Empty list is returned if there are quotation marks in the tokens, as phrase queries are not supported for free text search.
Invalid terms (i.e. not defined in dictionary in indexing) are removed to avoid KeyError.
For evaluation, postings list for each term are retrieved, used to calculate tf-idf scores following algorithm in lecture notes. Cosine normalization was not employed, as due to the nature of legal documents, it is possible that important terms e.g. court name might only occur a few times in a lengthy document.
Scores are returned in sorted order, documents with equal scores are sorted by document number.

process_query_boolean() works as follows: split the string into a list of tokens of operators/terms, remove punctuation and stem, process phrases to bigrams, convert the order of tokens to be in postfix notation, evaluate the query, loading postings from disk as needed.
Invalid terms (i.e. not defined in dictionary in indexing) are removed to avoid KeyError.
Bigrams are created by splitting tokens list on AND, with each item in between becoming an individual token list, and processed based on if it is a single term (no change), bigram (create 1 bigram, 2 single terms with OR operator), or trigram (create 2 bigrams, 3 single terms with OR operator)
AND and OR operations were both implemented for evaluation, with OR operations necessary for query expansion of boolean queries since expanding with AND would actually increase the specificity of the query instead.
Tf-idf scores were calculated in fetch_postings_lists during the postfix process. This allows documents to be sorted and returned based on this score, with documents with equal scores sorted by document number. 

search.py handles the processing of the query and output file by calling either process_query or process_query_boolean in QueryProcessor depending on if "AND" exists in the query string.

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

[Pandas](https://pandas.pydata.org/docs/user_guide/index.html#user-guide) - Pandas documentation for indexing of large CSV file.
