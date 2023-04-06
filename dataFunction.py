import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model
model = SentenceTransformer('all-minilm-l6-v2')

# Load dataset and create CountVectorizer object for sparse vectors
corpus = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.',
    'To be or not to be.']  # List of documents

vectorizer = CountVectorizer()
sparse_vectors = vectorizer.fit_transform(corpus)

print("SPARSE VECTORS: ")
print(sparse_vectors)

# Compute BM25 weights for each document (to be used as sparse vectors)
'''
doc_freqs = np.array(sparse_vectors.sum(axis=0))[0]
doc_lengths = np.array(sparse_vectors.sum(axis=1)).reshape(-1)
avg_doc_length = np.mean(doc_lengths)
k1 = 1.2
b = 0.75
idf = np.log((len(corpus) - doc_freqs + 0.5) / (doc_freqs + 0.5))
bm25_weights = ((sparse_vectors * (k1 + 1)) / (sparse_vectors + k1 * (1 - b + b * doc_lengths / avg_doc_length))) * idf
'''




def retrieve_top_k(query, k):
    # Compute inner product of query with all dense vectors
    dense_scores = np.dot(query[0], dense_vectors.T)

    # Compute inner product of query with all sparse vectors (BM25 weights)
    sparse_query = vectorizer.transform([query[1]])
    sparse_scores = np.dot(bm25_weights, sparse_query.T).flatten()

    # Merge the two scores and retrieve the top-k documents
    all_scores = dense_scores + sparse_scores
    top_k_indices = np.argsort(-all_scores)[:k]
    top_k_scores = all_scores[top_k_indices]
    top_k_documents = [corpus[i] for i in top_k_indices]

    return top_k_documents, top_k_scores
