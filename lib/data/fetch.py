from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np


def labelled_comments(dataset_path):
    raw_data = pd.read_csv(dataset_path).as_matrix()
    np.random.shuffle(raw_data)
    data_y = raw_data[:, 1]
    raw_x = raw_data[:, 0]
    data_x = list()
    for sentence in raw_x:
        data_x.append(word_tokenize(sentence))
    data_x = np.array(data_x)
    return data_x, data_y


def issue_comments():
    pass


def review_comments():
    pass