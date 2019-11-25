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

def topk_mean(m, k, inplace=False):
	n = m.shape[0]
	ans = np.zeros(n, dtype=m.dtype)
	if k <= 0:
		return ans

	if not inplace:
		m = np.array(m)

	ind0 = np.arange(n)
	ind1 = np.empty(n, dtype=int)
	minimum = m.min()
	for i in range(k):
		m.argmax(axis=1, out=ind1)
		ans += m[ind0, ind1]
		m[ind0, ind1] = minimum
	return ans / k


#loading seed dictionary for a better initialization
with open("./hi-en.tr") as f:
	aligned_words = f.readlines()

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

#using alligned words to create a better seed dictionary
# for i in aligned_words:
# 	hindi_word = i.strip().split()[1].strip()
# 	english_word = i.strip().split()[0].strip()

# 	if hindi_word in hindi_vocab and english_word in english_vocab:
# 		D[hindi_vocab[hindi_word], english_vocab[english_word]] = 1

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
sim = X1.dot(Y1.T)
knn_sim_fwd = topk_mean(sim, k = 7)
knn_sim_bwd = topk_mean(sim.T, k = 7)
sim -= knn_sim_fwd[:, np.newaxis]/2 + knn_sim_bwd/2

nearest_neighbors = sim.argmax(axis = 1)
D[range(nearest_neighbors.shape[0]), nearest_neighbors] = 1

#training loop
num_iter = 100
sample_size = 100
k = 0

for i in range(num_iter):
	#computing the optimal orthogonal matrix which maximizes the current dict D
	u, s, vh = np.linalg.svd(np.dot(X.T, np.dot(D, Y)))
	WX = u
	WY = vh

	#computing the optimal dictionary using the similarity matrix of the mapped embeddings
	sim = np.dot(X, WX).dot(np.dot(Y, WY).T)
	knn_sim_fwd = topk_mean(sim, k = 7)
	knn_sim_bwd = topk_mean(sim.T, k = 7)
	sim -= knn_sim_fwd[:, np.newaxis]/2 + knn_sim_bwd/2

	nearest_neighbors = sim.argmax(axis = 1)

	# temp = np.copy(D)
	D[range(nearest_neighbors.shape[0]), nearest_neighbors] = 1

	for j in aligned_words[k:k+sample_size]:
		hindi_word = j.strip().split()[0].strip()
		english_word = j.strip().split()[1].strip()

		if hindi_word in hindi_vocab and english_word in english_vocab:
			D[hindi_vocab[hindi_word], english_vocab[english_word]] = 1

	k += sample_size
	# print((D==temp).sum())

#final step
u, s, vh = np.linalg.svd(np.dot(X.T, np.dot(D, Y)), full_matrices = False)
s = np.diag(s)
WX = u.dot(np.sqrt(s))
WY = vh.dot(np.sqrt(s))

#testing
sim = np.dot(X, WX).dot(np.dot(Y, WY).T)
knn_sim_fwd = topk_mean(sim, k = 7)
knn_sim_bwd = topk_mean(sim.T, k = 7)
sim -= knn_sim_fwd[:, np.newaxis]/2 + knn_sim_bwd/2

nearest_neighbors = sim.argmax(axis = 1)

pickle.dump(np.dot(X, WX), open("./xwx_with_csls_with_improvement.pkl", "wb"))
pickle.dump(np.dot(Y, WY), open("./ywy_with_csls_with_improvement.pkl", "wb"))

# for i in hindi_idx2word:
# 	print(hindi_idx2word[i], english_idx2word[nearest_neighbors[i]])

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
