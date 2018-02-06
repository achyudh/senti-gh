from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec["a"])

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
        self.dim = len(word2vec["a"])

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class HybridEmbeddingVectorizer(object):
    def __init__(self, word2vec1, word2vec2, dim1, dim2):
        self.word2vec1 = word2vec1
        self.word2vec2 = word2vec2
        self.word2weight = None
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim = dim1 + dim2

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        result = list()
        for words in X:
            sent_vector = list()
            for w in words:
                if w in self.word2vec1 and w in self.word2vec2:
                    sent_vector.append(np.concatenate((self.word2vec1[w], self.word2vec2[w])))
                elif w in self.word2vec1 and w not in self.word2vec2:
                    sent_vector.append(np.concatenate((self.word2vec1[w], np.zeros(self.dim2))))
                elif w not in self.word2vec1 and w in self.word2vec2:
                    sent_vector.append(np.concatenate((np.zeros(self.dim1), self.word2vec2[w])))
                else:
                    sent_vector.append(np.zeros(self.dim))
            if len(sent_vector) > 0:
                result.append(np.mean(np.array(sent_vector), axis=0).reshape(600))
            else:
                result.append(np.zeros((600)))
        return np.array(result)
