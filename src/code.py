import pickle 
import numpy as np 

hindi_vocab = pickle.load(open("./hindi_vocab.pkl", "rb"))
hindi_embedding_matrix = pickle.load(open("./hindi_embedding_matrix.pkl", "rb"))
english_vocab = pickle.load(open("./english_vocab.pkl", "rb"))
english_embedding_matrix = pickle.load(open("./english_embedding_matrix.pkl", "rb"))

X = hindi_embedding_matrix
Y = english_embedding_matrix

print(np.dot(X, X.T).shape)
print(np.dot(Y, Y.T).shape)

