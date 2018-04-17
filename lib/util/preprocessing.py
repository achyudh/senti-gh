from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tinydb import TinyDB
import numpy as np


def user_ipa_count(dataset):
    result = dict()
    db = TinyDB(dataset)
    for entry in db:
        for thread in entry.values():
            for comment in thread['comments']:
                if 'ipa' in comment:
                    user_login = comment['user']['login']
                    category = comment['ipa']
                    if user_login not in result:
                        result[user_login] = dict()
                        result[user_login]['a'] = 0
                        result[user_login]['b'] = 0
                        result[user_login]['c'] = 0
                        result[user_login]['d'] = 0
                    if category != '-':
                        result[user_login][category.lower()] += 1
    for v in result.values():
        sum = v['a'] + v['b'] + v['c'] + v['d']
        if sum != 0:
            v['a'] = v['a']/sum
            v['b'] = v['b']/sum
            v['c'] = v['c']/sum
            v['d'] = v['d']/sum
    return result


def read_csv(path, headers=True):
    result = list()
    with open(path) as csv_file:
        if headers:
            _temp = csv_file.readline()
        for line in csv_file:
            if line.strip() is not "":
                split_line = [x.strip() for x in line.strip().split(',')]
                if split_line[0] is not "":
                    result.append(split_line)
    return result


def to_extended_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    for i in range(n):
        if y[i] != -1:
            categorical[i, y[i]] = 1
    return categorical


def make_network_ready(data, num_classes, tokenizer = None, max_sequence_len=400, enforce_max_len=False):
    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data[:,0])
    sequences = tokenizer.texts_to_sequences(data[:,0])
    seq_lengths = [len(seq) for seq in sequences]
    if not enforce_max_len:
        max_sequence_len = min(max_sequence_len, max(seq_lengths))
    data_x = pad_sequences(sequences, maxlen=max_sequence_len)
    data_y = [int(x) for x in data[:,1]]
    data_y_cat = to_categorical(data_y, num_classes=num_classes)
    return data_x, data_y_cat, tokenizer, max_sequence_len

if __name__ == '__main__':
    print(to_extended_categorical([1, 2, -1, 0, 1, 0, 2, 0, -1], 3))