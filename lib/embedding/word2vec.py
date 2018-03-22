from lib.util import mediawiki, ngram
from gensim.models import phrases
from lib.data import fetch
import numpy as np
import gensim


def train_tensorflow():
    pass


def load_tensorflow():
    pass


def train_gensim(X, size=50, min_count=5):
    """
    Train a Word2Vec model from scratch with Gensim
    :param X: A list of tokenized texts (i.e. list of lists of tokens)
    :return: A trained Word2Vec model
    """
    print("Training Word2Vec...")
    model = gensim.models.Word2Vec(X, size=size, workers=8, min_count=min_count)
    model.save('data/embedding/word2vec/gensim_size%s_min%s' % (size, min_count))
    return model


def load_gensim(model_path='data/embedding/word2vec/gensim_size50_min3', binary=False):
    if binary:
        return gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    else:
        return gensim.models.Word2Vec.load(model_path)


def embedding_matrix(word_index, model_path='data/embedding/word2vec/gensim_size50_min3', binary=False):
    if binary:
        size = int(model_path.split('.')[-2].split('/')[-1].split('_')[1][4:])
    else:
        size = int(model_path.split('/')[-1].split('_')[1][4:])
    w2v = load_gensim(model_path, binary)
    embedding_map = np.zeros((len(word_index) + 1, size))
    for word, i in word_index.items():
        if word in w2v:
            embedding_map[i] = w2v[word]
    return embedding_map


if __name__ == '__main__':
    # bigram_model, trigram_model = ngram.load()
    # bigram_phraser = phrases.Phraser(bigram_model)
    # trigram_phraser = phrases.Phraser(trigram_model)
    # token_matrix = np.concatenate((mediawiki.fetch(detect_ngrams=False), fetch.complete_text()))
    # w2v = train_gensim(trigram_phraser[bigram_phraser[token_matrix]])
    # w2v = load_gensim('data/embedding/word2vec/gensim_size50_min5')
    print("Custom:")
    print("Vocab. size:", len(w2v.wv.vocab))
    print('repo', [x[0] for x in w2v.most_similar('repo')])
    print('fork', [x[0] for x in w2v.most_similar('fork')])
    print('branch', [x[0] for x in w2v.most_similar('branch')])
    print('patch', [x[0] for x in w2v.most_similar('patch')])
    print('tree', [x[0] for x in w2v.most_similar('tree')])
    print('ssh', [x[0] for x in w2v.most_similar('ssh')])
    print('whitespace', [x[0] for x in w2v.most_similar('whitespace')])
    print('android', [x[0] for x in w2v.most_similar('android')])
    print('account', [x[0] for x in w2v.most_similar('account')])

    w2v = load_gensim('data/embedding/word2vec/gensim_size300_min5')
    print("Google:")
    print("Vocab. size:", len(w2v.wv.vocab))
    print('repo', [x[0] for x in w2v.most_similar('repo')])
    print('fork', [x[0] for x in w2v.most_similar('fork')])
    print('branch', [x[0] for x in w2v.most_similar('branch')])
    print('patch', [x[0] for x in w2v.most_similar('patch')])
    print('tree', [x[0] for x in w2v.most_similar('tree')])
    print('ssh', [x[0] for x in w2v.most_similar('ssh')])
    print('whitespace', [x[0] for x in w2v.most_similar('whitespace')])
    print('android', [x[0] for x in w2v.most_similar('android')])
    print('account', [x[0] for x in w2v.most_similar('account')])

