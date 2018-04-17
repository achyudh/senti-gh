from lib.embedding import word2vec, fasttext
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.layers import Dense, Input, Embedding, Bidirectional, LSTM, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from lib.util import preprocessing
import tensorflow as tf
import pandas as pd
import numpy as np


def train(train_x, train_y, evaluate_x, evaluate_y, embedding_map, embedding_dim, max_sequence_len, num_classes):
    with tf.device('/gpu:0'):
        sequence_input = Input(shape=(max_sequence_len,), dtype='int32')
        embedding_layer_1 = Embedding(len(tokenizer.word_index) + 1, embedding_dim, weights=[embedding_map],
                                input_length=max_sequence_len, trainable=False)
        embedded_sequences_1 = embedding_layer_1(sequence_input)
        l_lstm = Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2))(embedded_sequences_1)
        l_dense1 = Dense(50, activation='relu')(l_lstm)
        # l_dropout1 = Dropout(0.4)(l_dense1)
        preds = Dense(num_classes, activation='softmax')(l_dense1)
        lstm_model = Model(sequence_input, preds)
        lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        lstm_model.summary()
        early_stopping_callback = EarlyStopping(patience=2, monitor='val_loss')
        lstm_model.fit(train_x, train_y, validation_data=(evaluate_x, evaluate_y), epochs=20, batch_size=64,
                      callbacks=[early_stopping_callback])
    return lstm_model


def predict(classifier, predict_x):
    return classifier.predict(predict_x)


def evaluate(classifier, evaluate_x, evaluate_y):
    predict_y = predict(classifier, evaluate_x).argmax(axis=1)
    evaluate_y = evaluate_y.argmax(axis=1)
    return {"individual": precision_recall_fscore_support(evaluate_y, predict_y),
            "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}


def cross_val(data_x, data_y, embedding_map, embedding_dim, max_sequence_len, num_classes, n_splits=5):
    skf = StratifiedKFold(n_splits, random_state=157)
    print("Performing cross validation (%d-fold)..." % n_splits)
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    mean_accuracy = 0
    iteration = 0
    for train_index, test_index in skf.split(data_x, data_y.argmax(axis=1)):
        print("Iteration %d of %d" % (iteration, n_splits))
        iteration += 1
        cnn_pipeline = train(data_x[train_index], data_y[train_index], data_x[test_index], data_y[test_index], embedding_map, embedding_dim, max_sequence_len, num_classes)
        metrics = evaluate(cnn_pipeline, data_x[test_index], data_y[test_index])
        precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
        recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
        mean_accuracy += metrics['micro-average'][0]
        print("Precision, Recall, F_Score, Support")
        print(metrics)
    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/n_splits, [precision/n_splits for precision in precision_list], [recall/n_splits for recall in recall_list]))


if __name__ == '__main__':
    embedding_dim_1 = embedding_dim_2 = 300
    num_classes = 2
    data = pd.read_csv("data/labelled/Gerrit.csv").as_matrix()
    # data = pd.read_csv("data/labelled/StackOverflow.csv", encoding='latin1').as_matrix()
    data_x, data_y_cat, tokenizer, max_sequence_len = preprocessing.make_network_ready(data, num_classes)
    print("Dataset loaded to memory. Size:", len(data_y_cat))
    # embedding_map_1 = word2vec.embedding_matrix(tokenizer.word_index, model_path="data/embedding/word2vec/googlenews_size300.bin", binary=True)
    embedding_map_1 = word2vec.embedding_matrix(tokenizer.word_index)
    cross_val(data_x, data_y_cat, embedding_map_1, embedding_dim_1, max_sequence_len, num_classes, n_splits=10)
