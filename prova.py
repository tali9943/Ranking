import math
import collections
import nltk
from collections import defaultdict


def calculateIDF(tokenized_docs,term_freqs):
    
    #dictionary with the frequency of each term in all documents
    all_terms = [term for doc in tokenized_docs for term in doc]
    
    df = collections.Counter(all_terms)

    # Calculate IDF for each term
    N = len(tokenized_docs)
    
    idf = {}
    
    for term in df:
        n = len([doc for doc in tokenized_docs if term in doc])
        idf[term] = math.log(N / float(df[term] + 1)) #math.log(1 + (N - n + 0.5)/(n + 0.5))
    return idf




def calculateIDFBM25(tokenized_docs):
    #dictionary with the frequency of each term in all documents
    all_terms = [term for doc in tokenized_docs for term in doc]
    
    df = defaultdict(int)

    # Calculate DF for each term
    for term in all_terms:
        df[term] += 1
    # Calculate IDF for each term
    N = len(tokenized_docs)
    
    idf = {}
    
    for term in df:
        n = len([doc for doc in tokenized_docs if term in doc])
        idf[term] = math.log((N - n + 0.5)/(n + 0.5)+1)
    
    return idf




#idf = calculateIDFBM25(tokenized_docs)
idf = calculateIDF(tokenized_docs,term_freqs)



def calculateTF(tokenized_docs):
    term_freqs = []
    for doc in tokenized_docs:
        doc_freq = collections.Counter(doc) #number of repetition for each word
       
        total_terms = len(doc) #length for each document
        
        for term in doc_freq:
            doc_freq[term] /= float(total_terms)
        term_freqs.append(doc_freq)

    return term_freqs



def calculateTF2(tokenized_docs):
    term_freqs = []
    for doc in tokenized_docs:
        doc_freq = {}
        total_terms = len(doc)
        for term in doc:
            doc_freq[term] = doc_freq.get(term, 0) + 1
        for term in doc_freq:
            doc_freq[term] /= float(total_terms)
        term_freqs.append(doc_freq)

    return term_freqs



def calculateTFbm25(tokenized_docs):
    term_freqs = []
    k1 = 1.5 # parameter for controlling term frequency normalization
    b = 0.75 # parameter for controlling document length normalization
    avgdl = sum(len(doc) for doc in tokenized_docs) / len(tokenized_docs) # average document length

    for doc in tokenized_docs:
        doc_freq = collections.Counter(doc) #number of repetition for each word
        total_terms = len(doc) #length for each document
        doc_len_norm = ((1 - b) + b * (total_terms / avgdl)) # document length normalization factor

        for term in doc_freq:
            tf = doc_freq[term] / total_terms # term frequency
            tf_norm = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (total_terms / avgdl)) + tf) # normalized term frequency with BM25 weighting
            doc_freq[term] = tf_norm

        term_freqs.append(doc_freq)

    return term_freqs

    
#term_freqs = calculateTF2(tokenized_docs)
term_freqs = calculateTFbm25(tokenized_docs)

