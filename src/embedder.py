import pickle
from pprint import pprint as print

from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath

EMBEDDING_DIM = 100

corpus_hi = datapath('../data/train.hi')
corpus_en = datapath('../data/train.en')

modelhi = FT_gensim(size = EMBEDDING_DIM)
modelen = FT_gensim(size = EMBEDDING_DIM)

modelhi.build_vocabulary(corpus_file=corpus_hi)
modelen.build_vocabulary(corpus_file=corpus_en)

modelhi.train(
        corpus_file=corpus_hi,
        epochs=modelhi.epochs,
        total_examples=modelhi.corpus_count,
        total_words=modelhi.corpus_total_words
)
modelen.train(
        corpus_file=corpus_en,
        epochs=modelen.epochs,
        total_examples=modelen.corpus_count,
        total_words=modelen.corpus_total_words
)


