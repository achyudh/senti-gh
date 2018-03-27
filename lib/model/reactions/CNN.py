from lib.embedding.word2vec import embedding_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.layers import Dense, Input, Flatten, Embedding
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dropout, GlobalMaxPool1D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from gensim.models import phrases
from nltk.tokenize import word_tokenize
from lib.data import fetch
from lib.util import preprocessing, ngram
import tensorflow as tf
import pandas as pd
import numpy as np


def train(train_x, train_y, evaluate_x, evaluate_y, embedding_map, embedding_dim, max_sequence_len, num_classes):
    with tf.device('/gpu:1'):
        embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_map],
                                input_length=max_sequence_len, trainable=False)
        sequence_input = Input(shape=(max_sequence_len,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        l_conv1= Conv1D(200, 3, activation='relu', padding='valid',)(embedded_sequences)
        l_pool1 = MaxPooling1D(5)(l_conv1)
        l_conv2 = Conv1D(80, 3, activation='relu')(l_pool1)
        l_pool3 = GlobalMaxPool1D()(l_conv2)
        l_dense1 = Dense(60, activation='relu')(l_pool3)
        l_dropout1 = Dropout(0.2)(l_dense1)
        l_dense2 = Dense(20, activation='relu')(l_dropout1)
        l_dropout2 = Dropout(0.2)(l_dense2)
        preds = Dense(num_classes, activation='sigmoid')(l_dropout2)
        cnn_model = Model(sequence_input, preds)
        cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        cnn_model.summary()
        cnn_model.fit(train_x, train_y, validation_data=(evaluate_x, evaluate_y), epochs=20, batch_size=128)
    return cnn_model


def predict(classifier, predict_x):
    return classifier.predict(predict_x)


def evaluate(classifier, evaluate_x, evaluate_y):
    predict_y = predict(classifier, evaluate_x).argmax(axis=1)
    evaluate_y = evaluate_y.argmax(axis=1)
    return {"individual": precision_recall_fscore_support(evaluate_y, predict_y),
            "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}


def cross_val(data_x, data_y, embedding_map, embedding_dim, max_sequence_len, num_classes, n_splits=5):
    skf = StratifiedKFold(n_splits)
    print("Performing cross validation (%d-fold)..." % n_splits)
    mean_precision = 0
    mean_recall = 0
    for train_index, test_index in skf.split(data_x, data_y.argmax(axis=1)):
        cnn_pipeline = train(data_x[train_index], data_y[train_index], data_x[test_index], data_y[test_index], embedding_map, embedding_dim, max_sequence_len, num_classes)
        metrics = evaluate(cnn_pipeline, data_x[test_index], data_y[test_index])
        mean_precision += metrics['micro-average'][0]
        mean_recall += metrics['micro-average'][1]
        print("Precision, Recall, F_Score, Support")
        print(metrics)
    print("Mean precision: %s, Mean recall: %s" % (mean_precision/n_splits, mean_recall/n_splits))


if __name__ == '__main__':
    embedding_dim = 300
    num_classes = 2
    data = pd.read_csv("data/labelled/JIRA.csv").as_matrix()
    bigram_model, trigram_model = ngram.load()
    data_x = np.array(([word_tokenize(x.lower()) for x in data[:,0]]))
    data_y = [int(x) for x in data[:,1]]
    # data_x, reaction_matrix = fetch.sentences_with_reactions("data/user", tokenize=False)
    # data_y = reaction_matrix[:, 0]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data[:,0])
    sequences = tokenizer.texts_to_sequences(data[:,0])
    max_sequence_len = max(len(seq) for seq in sequences)
    max_sequence_len = min(500, max_sequence_len)
    data_x = pad_sequences(sequences, maxlen=max_sequence_len)
    data_y_cat = to_categorical(data_y, num_classes=num_classes)
    word_index = tokenizer.word_index
    embedding_map = embedding_matrix(word_index, model_path='data/embedding/word2vec/gensim_size300_min5')
    print(data_x)
    print(data_y_cat)
    cross_val(data_x, data_y_cat, embedding_map, embedding_dim, max_sequence_len, num_classes, n_splits=5)