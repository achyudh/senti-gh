from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nltk import tokenize
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


def make_hierarchical_network_ready(data, num_classes, tokenizer=None, max_sequence_len=200, max_sequences=20,
                                    enforce_max_len=False, filter_words=False):
    temp_data = list()
    for seq in data[:, 0]:
        temp_data.append(' '.join(seq.split()))
    if tokenizer is None:
        tokenizer = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(temp_data)

    raw_data = list()
    max_sequences_actual = -1
    max_sequence_len_actual = -1
    for seq in data[:,0]:
        sentences = tokenize.sent_tokenize(seq)
        raw_data.append(sentences)
        max_sequences_actual = max(len(sentences), max_sequences_actual)
        for sentence in sentences:
            word_tokens = text_to_word_sequence(sentence, filters='!"#$%&()*+,./:;<=>?@[\]^_`{|}~', lower=True)
            max_sequence_len_actual = max(len(word_tokens), max_sequence_len_actual)

    if not enforce_max_len:
            max_sequence_len = min(max_sequence_len, max_sequence_len_actual)
            max_sequences = min(max_sequences, max_sequences_actual)

    data_x = np.zeros((len(data), max_sequences, max_sequence_len), dtype='int32')
    print("Max. Seq. Length: %d; Max Seq.: %d" %(max_sequence_len, max_sequences))

    index_filter = set()
    if filter_words:
        for word, i in tokenizer.word_index.items():
            if not (word.isalpha() or "'" in word or "-" in word):
                index_filter.add(i)

    for i, sentences in enumerate(raw_data):
        for j, sentence in enumerate(sentences):
            if j < max_sequences:
                k = 0
                word_tokens = text_to_word_sequence(' '.join(sentence.split()), filters='!"#$%&()*+,./:;<=>?@[\]^_`{|}~', lower=True)
                for word in word_tokens:
                    if k < max_sequence_len:
                        if word in tokenizer.word_index:
                            if not filter_words or tokenizer.word_index[word] not in index_filter:
                                    data_x[i, j, k] = tokenizer.word_index[word]
                        k = k + 1

    data_y = [int(x) for x in data[:, 1]]
    data_y_cat = to_categorical(data_y, num_classes=num_classes)
    return data_x, data_y_cat, tokenizer, max_sequence_len, max_sequences


def make_network_ready(data, num_classes, tokenizer=None, max_sequence_len=400, enforce_max_len=False, filter_words=False):
    if tokenizer is None:
        tokenizer = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(data[:,0])
    raw_sequences = tokenizer.texts_to_sequences(data[:,0])

    if filter_words:
        sequences = list()
        index_filter = set()
        for word, i in tokenizer.word_index.items():
            if not (word.isalpha() or "'" in word or "-" in word):
                index_filter.add(i)
        for seq in raw_sequences:
            new_seq = list()
            for i in seq:
                if i not in index_filter:
                    new_seq.append(i)
            sequences.append(new_seq)
    else:
        sequences = raw_sequences

    seq_lengths = [len(seq) for seq in sequences]
    if not enforce_max_len:
        max_sequence_len = min(max_sequence_len, max(seq_lengths))
    data_x = pad_sequences(sequences, maxlen=max_sequence_len)
    data_y = [int(x) for x in data[:, 1]]
    data_y_cat = to_categorical(data_y, num_classes=num_classes)
    return data_x, data_y_cat, tokenizer, max_sequence_len


if __name__ == '__main__':
    print(to_extended_categorical([1, 2, -1, 0, 1, 0, 2, 0, -1], 3))