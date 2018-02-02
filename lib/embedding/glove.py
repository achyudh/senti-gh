import numpy as np


def load_glove6B():
    """

    :param X: A list of tokenized texts (i.e. list of lists of tokens)
    :return: A pre-trained Word2Vec model on 6B tokens
    """
    with open("data/glove/glove.6B.50d.txt", "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    print("Glove6B loaded into memory")
    return w2v


def load_glove840B():
    """

    :param X:
    :return: A pre-trained Word2Vec model on 840B tokens
    """
    with open("data/glove/glove.840B.300d.txt", "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    return w2v
