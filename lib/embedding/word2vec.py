import gensim


def train_tensorflow():
    pass


def train_gensim(X, size=200, min_count=5):
    """
    Train a Word2Vec model from scratch with Gensim
    :param X: A list of tokenized texts (i.e. list of lists of tokens)
    :return: A trained Word2Vec model
    """
    print("Training Word2Vec...")
    model = gensim.models.Word2Vec(X, size=size, workers=8, min_count=min_count)
    model.save('data/embedding/word2vec/gensim_size%s_min%s' % (size, min_count))
    return model


def load_gensim(model_path='data/embedding/word2vec/gensim_size300_min5'):
    return gensim.models.Word2Vec.load(model_path)