import pickle
import fastText as ft
import numpy as np 

MIN_FREQ = 5
EMBEDDING_DIM = 100

# creating vocabulary from the data
def create_vocab(data):
	vocab = {}
	for sent in data:
		for word in sent.strip().split():
			if word.strip() not in vocab:
				vocab[word.strip()] = 1
			else:
				vocab[word.strip()] += 1
	return vocab

# Removing elements from vocab which have frequency less than min frequency and constructing embedding matrix
def create_embedding_matrix(vocab, model):
	idx = 0
	new_vocab = {}
	embedding_matrix = []

	for word in vocab:
		if vocab[word] > MIN_FREQ:
			new_vocab[word] = idx
			embedding_matrix.append(model.get_word_vector(word))
			idx += 1

	embedding_matrix = np.array(embedding_matrix)

	return new_vocab, embedding_matrix

# training fastText embeddings for the data
print("Training fastText embeddings on hindi data...")
model1 = ft.train_unsupervised("./train.hi", model = "cbow", dim = EMBEDDING_DIM)
print("Training fastText embeddings on english data...")
model2 = ft.train_unsupervised("./train.en", model = "cbow", dim = EMBEDDING_DIM)

# reading data
with open("./train.hi") as f:
	hindi_data = f.readlines()

with open("./train.en") as f:
	english_data = f.readlines()

hindi_vocab = create_vocab(hindi_data)
english_vocab = create_vocab(english_data)

reduced_hindi_vocab, hindi_embedding_matrix = create_embedding_matrix(hindi_vocab, model1)
reduced_english_vocab, english_embedding_matrix = create_embedding_matrix(english_vocab, model2)

print("Saving the vocabulary and embedding matrix.")

pickle.dump(reduced_hindi_vocab, open("./hindi_vocab.pkl", "wb"))
pickle.dump(hindi_embedding_matrix, open("./hindi_embedding_matrix.pkl", "wb"))
pickle.dump(reduced_english_vocab, open("./english_vocab.pkl", "wb"))
pickle.dump(english_embedding_matrix, open("./english_embedding_matrix.pkl", "wb"))
