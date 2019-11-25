import pickle
import numpy as np
import matplotlib.pyplot as plt 

#Normalizing the sorted_MX and sorted_MY
def normalize(X):
	#length normalization
	x = X/np.linalg.norm(X, axis = 1).reshape(-1, 1)
	#mean centering each dimension
	x = x - x.mean(axis = 0)
	#length normalizing the mean centered data
	x = x/np.linalg.norm(x, axis = 1).reshape(-1, 1)
	return x

# computing sqaure root of similiarity matrices for both the embedding matrices using svd
def sim_matrix(X, size):
	u, s, _ = np.linalg.svd(X[:size], full_matrices = False)
	return np.dot(u*s, u.T)

def distance_matrix(X1, Y1):
	norm_X1 = np.linalg.norm(X1, axis = 1).reshape(-1,1)**2
	b = np.ones((norm_X1.shape[0], norm_X1.shape[1]+1))
	b[:, :-1] = norm_X1
	norm_X1 = b

	norm_Y1 = np.linalg.norm(Y1, axis = 1).reshape(-1,1)**2
	b = np.ones((norm_Y1.shape[0], norm_Y1.shape[1]+1))
	b[:, 1:] = norm_Y1
	norm_Y1 = b

	dot_p = -2*np.matmul(X1, Y1.T)
	d_matrix = np.matmul(norm_X1, norm_Y1.T) + dot_p
	return d_matrix

def create_idx2word_dict(vocab):
	idx2word = {}
	for i in vocab:
		idx2word[vocab[i]] = i

	return idx2word

# loading hindi english vocabulary and embedding matrices
hindi_embedding_matrix = pickle.load(open("./hindi_embedding_matrix.pkl", "rb"))
english_embedding_matrix = pickle.load(open("./english_embedding_matrix.pkl", "rb"))
hindi_vocab = pickle.load(open("./hindi_vocab.pkl", "rb"))
english_vocab = pickle.load(open("./english_vocab.pkl", "rb"))

hindi_idx2word = create_idx2word_dict(hindi_vocab)
english_idx2word = create_idx2word_dict(english_vocab)

X = hindi_embedding_matrix
Y = english_embedding_matrix
D = np.zeros((X.shape[0], Y.shape[0]))

with open("./hi-en.tr") as f:
	aligned_words = f.readlines()

SIM_SIZE = min(X.shape[0], Y.shape[0])

#normalizing the embeddings
X = normalize(X)
Y = normalize(Y)

#similarity matrices
MX = sim_matrix(X, SIM_SIZE)
MY = sim_matrix(Y, SIM_SIZE)

#sorting MX and MY along the rows in order to align them along the j dimension
sorted_MX = np.sort(MX, axis = 1)
sorted_MY = np.sort(MY, axis = 1)

#normalized embeddings for extracting alignment across vocabulary(ith dimension)
X1 = normalize(sorted_MX)
Y1 = normalize(sorted_MY)

#constructing initial solution for the iterative algorithm
d_matrix = distance_matrix(X1, Y1)
nearest_neighbors = d_matrix.argmin(axis = 1)
D[range(nearest_neighbors.shape[0]), nearest_neighbors] = 1

#training loop
num_iter = 500

sample_size = 50
k = 0

for i in range(num_iter):
	#computing the optimal orthogonal matrix which maximizes the current dict D
	u, s, vh = np.linalg.svd(np.dot(X.T, np.dot(D, Y)))
	WX = u
	WY = vh

	#computing the optimal dictionary using the similarity matrix of the mapped embeddings
	d_matrix = distance_matrix(np.dot(X, WX), np.dot(Y, WY))
	nearest_neighbors = d_matrix.argmin(axis = 1)

	D[range(nearest_neighbors.shape[0]), nearest_neighbors] = 1

	for j in aligned_words[k:k+sample_size]:
		hindi_word = j.strip().split()[0].strip()
		english_word = j.strip().split()[1].strip()

		if hindi_word in hindi_vocab and english_word in english_vocab:
			D[hindi_vocab[hindi_word], english_vocab[english_word]] = 1

	k += sample_size

#final step
u, s, vh = np.linalg.svd(np.dot(X.T, np.dot(D, Y)))
WX = u*s
WY = vh*s

#testing
d_matrix = distance_matrix(np.dot(X, WX), np.dot(Y, WY))
nearest_neighbors = d_matrix.argmin(axis = 1)

pickle.dump(np.dot(X, WX), open("./xwx_without_csls_with_improvement.pkl", "wb"))
pickle.dump(np.dot(Y, WY), open("./ywy_without_csls_with_improvement.pkl", "wb"))

with open("hi-en.te") as f:
	data = f.readlines()

temp = 0
for line in data:
	if temp >= 1:
		break

	hi, en = line.split()
	hi = hi.strip()
	en = en.strip()

	try:
		hi_index = hindi_vocab[hi]
		en_index = english_vocab[en]
		hi_mean = sorted_MX[hi_index]
		en_mean = sorted_MY[en_index]

	except:
		continue	

	plt.subplot(1,2,1)
	plt.title('Distribution for {0}'.format(hi))
	plt.plot(hi_mean)
	
	plt.subplot(1,2,2)
	plt.title('Distribution for {0}'.format(en))
	plt.plot(en_mean)
	
	temp += 1

	plt.show()
