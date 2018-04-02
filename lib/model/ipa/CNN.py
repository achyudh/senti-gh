from lib.embedding import word2vec, fasttext
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.layers import Dense, Input, Flatten, Embedding, Concatenate
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dropout, GlobalMaxPool1D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from gensim.models import phrases
from nltk.tokenize import word_tokenize
from tinydb import TinyDB
from lib.data import fetch
from lib.util import preprocessing, ngram
import tensorflow as tf
import pandas as pd
import numpy as np


def train_dual(train_x, train_y, evaluate_x, evaluate_y, embedding_map_1, embedding_map_2, embedding_dim_1, embedding_dim_2, max_sequence_len, num_classes):
    with tf.device('/gpu:1'):
        sequence_input = Input(shape=(max_sequence_len,), dtype='int32')
        l_embedding1 = Embedding(len(word_index) + 1, embedding_dim_1, weights=[embedding_map_1],
                                input_length=max_sequence_len, trainable=False)
        embedded_sequences_1 = l_embedding1(sequence_input)
        l_conv11 = Conv1D(250, 10, activation='relu', padding='valid', )(embedded_sequences_1)
        l_pool11 = MaxPooling1D(5)(l_conv11)
        l_conv12 = Conv1D(150, 5, activation='relu')(l_pool11)
        l_pool12 = GlobalMaxPool1D()(l_conv12)

        l_embedding2 = Embedding(len(word_index) + 1, embedding_dim_2, weights=[embedding_map_2],
                                input_length=max_sequence_len, trainable=False)
        embedded_sequences_2 = l_embedding2(sequence_input)
        l_conv21 = Conv1D(200, 10, activation='relu', padding='valid', )(embedded_sequences_2)
        l_pool21 = MaxPooling1D(5)(l_conv21)
        l_conv22 = Conv1D(120, 5, activation='relu')(l_pool21)
        l_pool22 = GlobalMaxPool1D()(l_conv22)

        l_concat1 = Concatenate()([l_pool12, l_pool22])
        l_dense1 = Dense(120, activation='relu')(l_concat1)
        l_dropout1 = Dropout(0.2)(l_dense1)
        l_dense2 = Dense(60, activation='relu')(l_dropout1)
        l_dropout2 = Dropout(0.2)(l_dense2)
        l_dense3 = Dense(20, activation='relu')(l_dropout2)
        l_dropout3 = Dropout(0.2)(l_dense3)
        preds = Dense(num_classes, activation='sigmoid')(l_dropout3)

        cnn_model = Model(sequence_input, preds)
        cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        cnn_model.summary()
        cnn_model.fit(train_x, train_y, validation_data=(evaluate_x, evaluate_y), epochs=8, batch_size=128)
    return cnn_model


def train(train_x, train_y, evaluate_x, evaluate_y, embedding_map, embedding_dim, max_sequence_len, num_classes):
    with tf.device('/gpu:1'):
        embedding_layer_1 = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_map],
                                input_length=max_sequence_len, trainable=False)
        sequence_input = Input(shape=(max_sequence_len,), dtype='int32')
        embedded_sequences_1 = embedding_layer_1(sequence_input)
        l_conv1= Conv1D(250, 10, activation='relu', padding='valid',)(embedded_sequences_1)
        l_pool1 = MaxPooling1D(5)(l_conv1)
        l_conv2 = Conv1D(150, 5, activation='relu')(l_pool1)
        l_pool3 = GlobalMaxPool1D()(l_conv2)
        l_dense1 = Dense(80, activation='relu')(l_pool3)
        l_dropout1 = Dropout(0.2)(l_dense1)
        l_dense2 = Dense(20, activation='relu')(l_dropout1)
        l_dropout2 = Dropout(0.2)(l_dense2)
        preds = Dense(num_classes, activation='sigmoid')(l_dropout2)
        cnn_model = Model(sequence_input, preds)
        cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        cnn_model.summary()
        cnn_model.fit(train_x, train_y, validation_data=(evaluate_x, evaluate_y), epochs=8, batch_size=128)
    return cnn_model


def predict(classifier, predict_x):
    return classifier.predict(predict_x)


def evaluate(classifier, evaluate_x, evaluate_y):
    predict_y = predict(classifier, evaluate_x).argmax(axis=1)
    evaluate_y = evaluate_y.argmax(axis=1)
    return {"individual": precision_recall_fscore_support(evaluate_y, predict_y),
            "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}


def cross_val_dual(data_x, data_y, embedding_map_1, embedding_map_2, embedding_dim_1, embedding_dim_2, max_sequence_len, num_classes, n_splits=5):
    skf = StratifiedKFold(n_splits)
    print("Performing cross validation (%d-fold)..." % n_splits)
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    mean_accuracy = 0
    for train_index, test_index in skf.split(data_x, data_y.argmax(axis=1)):
        cnn_pipeline = train_dual(data_x[train_index], data_y[train_index], data_x[test_index], data_y[test_index], embedding_map_1, embedding_map_2, embedding_dim_1, embedding_dim_2, max_sequence_len, num_classes)
        metrics = evaluate(cnn_pipeline, data_x[test_index], data_y[test_index])
        precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
        recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
        mean_accuracy += metrics['micro-average'][0]
        print("Precision, Recall, F_Score, Support")
        print(metrics)
    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/n_splits, [precision/n_splits for precision in precision_list], [recall/n_splits for recall in recall_list]))


def cross_val(data_x, data_y, embedding_map, embedding_dim, max_sequence_len, num_classes, n_splits=5):
    skf = StratifiedKFold(n_splits)
    print("Performing cross validation (%d-fold)..." % n_splits)
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    mean_accuracy = 0
    for train_index, test_index in skf.split(data_x, data_y.argmax(axis=1)):
        cnn_pipeline = train(data_x[train_index], data_y[train_index], data_x[test_index], data_y[test_index], embedding_map, embedding_dim, max_sequence_len, num_classes)
        metrics = evaluate(cnn_pipeline, data_x[test_index], data_y[test_index])
        precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
        recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
        mean_accuracy += metrics['micro-average'][0]
        print("Precision, Recall, F_Score, Support")
        print(metrics)
    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/n_splits, [precision/n_splits for precision in precision_list], [recall/n_splits for recall in recall_list]))


if __name__ == '__main__':
    embedding_dim_1 =  embedding_dim_2 = 300
    num_classes = 4
    dataset_list = ['data/labelled/pull_requests/spring.json', 'data/labelled/pull_requests/oni.json', 'data/labelled/pull_requests/tensorflow.json']
    label_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    data_xs = set()
    data_x = list()
    data_y = list()
    for dataset_path in dataset_list:
        db = TinyDB(dataset_path)
        for entry in db:
            for thread in entry.values():
                for comment in thread['comments']:
                    if 'ipa' in comment and comment['ipa'] != '-':
                        comment_body = comment['body'].lower()
                        if comment_body not in data_xs:
                            data_xs.add(comment_body)
                            data_x.append(comment_body)
                            data_y.append(label_map[comment['ipa'].lower()])

    data_x = np.array(data_x)
    print(len(data_y), len(data_x))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data_x)
    sequences = tokenizer.texts_to_sequences(data_x)
    max_sequence_len = max(len(seq) for seq in sequences)
    max_sequence_len = min(500, max_sequence_len)
    data_x = pad_sequences(sequences, maxlen=max_sequence_len)
    data_y_cat = to_categorical(data_y, num_classes=num_classes)
    word_index = tokenizer.word_index
    embedding_map_1 = word2vec.embedding_matrix(word_index, model_path="data/embedding/word2vec/googlenews_size300.bin", binary=True)
    embedding_map_2 = word2vec.embedding_matrix(word_index)
    cross_val(data_x, data_y_cat, embedding_map_2, embedding_dim_2,max_sequence_len, num_classes, n_splits=10)
    # cross_val_dual(data_x, data_y_cat, embedding_map_1, embedding_map_2, embedding_dim_1, embedding_dim_2,max_sequence_len, num_classes, n_splits=10)