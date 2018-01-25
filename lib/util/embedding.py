from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
import gensim


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # If a text is empty we should return a vector of zeros with the same dimensionality as all the other vectors
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def word2vec(X):
    """
    Train a Word2Vec model from scratch with Gensim
    :param X: A list of tokenized texts (i.e. list of lists of tokens)
    :return: A trained Word2Vec model
    """
    model = gensim.models.Word2Vec(X, size=100)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    return w2v


def glove6B():
    """

    :param X: A list of tokenized texts (i.e. list of lists of tokens)
    :return: A pre-trained Word2Vec model on 6B tokens
    """
    with open("data/glove/glove.6B.300d.txt", "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    print("Glove6B loaded into memory")
    return w2v


def glove840B():
    """

    :param X:
    :return: A pre-trained Word2Vec model on 840B tokens
    """
    with open("data/glove/glove.840B.300d.txt", "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    return w2v
