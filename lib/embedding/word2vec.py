import gensim


def train_tensorflow():
    pass


def train_gensim(X, size=200):
    """
    Train a Word2Vec model from scratch with Gensim
    :param X: A list of tokenized texts (i.e. list of lists of tokens)
    :return: A trained Word2Vec model
    """
    print("Training Word2Vec...")
    model = gensim.models.Word2Vec(X, size=size, workers=8, min_count=5)
    model.save('data/embedding/word2vec/gensim')
    return model


def load_gensim(model_path='data/embedding/word2vec/gensim'):
    return gensim.models.Word2Vec.load(model_path)