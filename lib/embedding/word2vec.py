import gensim
import numpy as np


def train_tensorflow():
    pass


def train_gensim(X, size=50, min_count=3):
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
        return gensim.models.Word2Vec.load_word2vec_format(model_path, binary=True)
    else:
        return gensim.models.Word2Vec.load(model_path)


def embedding_matrix(word_index, model_path='data/embedding/word2vec/gensim_size50_min3', binary=False):
    size = int(model_path.split('/')[-1].split('_')[1][4:])
    w2v = load_gensim(model_path, binary)
    embedding_map = np.zeros((len(word_index) + 1, size))
    for word, i in word_index.items():
        embedding_vector = w2v.get(word, None)
        if embedding_vector is not None:
            embedding_map[i] = embedding_vector
    return embedding_map