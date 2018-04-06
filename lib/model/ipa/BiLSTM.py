from lib.embedding import word2vec, fasttext
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.layers import Dense, Input, Embedding, Bidirectional, LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tinydb import TinyDB
import tensorflow as tf
import numpy as np


def train(train_x, train_y, evaluate_x, evaluate_y, embedding_map, embedding_dim, max_sequence_len, num_classes):
    with tf.device('/gpu:1'):
        embedding_layer_1 = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_map],
                                      input_length=max_sequence_len, trainable=False)
        sequence_input = Input(shape=(max_sequence_len,), dtype='int32')
        embedded_sequences_1 = embedding_layer_1(sequence_input)
        l_lstm = Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2))(embedded_sequences_1)
        preds = Dense(num_classes, activation='softmax')(l_lstm)
        cnn_model = Model(sequence_input, preds)
        cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        cnn_model.summary()
        early_stopping_callback = EarlyStopping(patience=2, monitor='val_acc')
        cnn_model.fit(train_x, train_y, validation_data=(evaluate_x, evaluate_y), epochs=10, batch_size=64,
                      callbacks=[early_stopping_callback])
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
    embedding_dim_1 = embedding_dim_2 = 300
    num_classes = 4
    dataset_list = ['data/labelled/pull_requests/spring.json', 'data/labelled/pull_requests/oni.json', 'data/labelled/pull_requests/tensorflow.json']
    label_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    data_x = list()
    data_y = list()
    for dataset_path in dataset_list:
        db = TinyDB(dataset_path)
        for entry in db:
            for thread in entry.values():
                for comment in thread['comments']:
                    if 'ipa' in comment and comment['ipa'] != '-':
                        comment_body = comment['body'].lower()
                        data_x.append(comment_body)
                        data_y.append(label_map[comment['ipa'].lower()])

    data_x = np.array(data_x)
    print(len(data_y), len(data_x))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data_x)
    sequences = tokenizer.texts_to_sequences(data_x)
    max_sequence_len = max(len(seq) for seq in sequences)
    max_sequence_len = min(200, max_sequence_len)
    data_x = pad_sequences(sequences, maxlen=max_sequence_len)
    data_y_cat = to_categorical(data_y, num_classes=num_classes)
    word_index = tokenizer.word_index
    embedding_map_1 = word2vec.embedding_matrix(word_index, model_path="data/embedding/word2vec/googlenews_size300.bin", binary=True)
    # embedding_map_2 = word2vec.embedding_matrix(word_index)
    cross_val(data_x, data_y_cat, embedding_map_1, embedding_dim_1, max_sequence_len, num_classes, n_splits=10)
