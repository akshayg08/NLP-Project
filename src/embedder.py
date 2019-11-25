import os
import pickle
import numpy as np
from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath

EMBEDDING_DIM = 100
MIN_FREQ = 5

def get_embed(path, name):
    print('Loading data from {0}'.format(path))
    corpus = datapath(os.path.abspath(path))
    model = FT_gensim(size = EMBEDDING_DIM, min_count=MIN_FREQ)
    
    print('Building a vocabulary for the data')
    model.build_vocab(corpus_file=corpus)

    print('Training fastText embeddings for the data')
    model.train(corpus_file=corpus, epochs=model.epochs, total_examples=model.corpus_count, total_words=model.corpus_total_words)

    #print('Saving the model into {0}'.format(name))
    #model.save(name)
    
    new_vocab = {}
    embedding_matrix = []
    idx = 0

    print('Creating Word embedding matrix')
    for word in model.wv.vocab:
        new_vocab[word] = idx
        embedding_matrix.append(model.wv[word])
        idx += 1

    embedding_matrix = np.array(embedding_matrix)

    return new_vocab, embedding_matrix

reduced_hindi_vocab, hindi_embedding_matrix = get_embed('../data/train.hi', 'hi_mod.ft')
reduced_english_vocab, english_embedding_matrix = get_embed('../data/train.en', 'en_mod.ft')

print("Saving the vocabulary and embedding matrix.")

pickle.dump(reduced_hindi_vocab, open("./hindi_vocab.pkl", "wb"))
pickle.dump(hindi_embedding_matrix, open("./hindi_embedding_matrix.pkl", "wb"))
pickle.dump(reduced_english_vocab, open("./english_vocab.pkl", "wb"))
pickle.dump(english_embedding_matrix, open("./english_embedding_matrix.pkl", "wb"))

