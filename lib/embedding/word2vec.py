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
    w2v = load_gensim()
    print("Custom:")
    print("Vocab. size:", len(w2v.wv.vocab))
    # print('repo', [x[0] for x in w2v.most_similar('repo')])
    # print('fork', [x[0] for x in w2v.most_similar('fork')])
    # print('branch', [x[0] for x in w2v.most_similar('branch')])
    # print('patch', [x[0] for x in w2v.most_similar('patch')])
    # print('tree', [x[0] for x in w2v.most_similar('tree')])
    # print('ssh', [x[0] for x in w2v.most_similar('ssh')])
    # print('rails', [x[0] for x in w2v.most_similar('rails')])
    print('android', [x[0] for x in w2v.most_similar('android')])
    print('call', [x[0] for x in w2v.most_similar('call')])

    w2v = load_gensim('data/embedding/word2vec/googlenews_size300.bin', binary=True)
    print("Google:")
    print("Vocab. size:", len(w2v.wv.vocab))
    # print('repo', [x[0] for x in w2v.most_similar('repo')])
    # print('fork', [x[0] for x in w2v.most_similar('fork')])
    # print('branch', [x[0] for x in w2v.most_similar('branch')])
    # print('patch', [x[0] for x in w2v.most_similar('patch')])
    # print('tree', [x[0] for x in w2v.most_similar('tree')])
    # print('ssh', [x[0] for x in w2v.most_similar('ssh')])
    # print('rails', [x[0] for x in w2v.most_similar('rails')])
    print('android', [x[0] for x in w2v.most_similar('android')])
    print('call', [x[0] for x in w2v.most_similar('call')])

