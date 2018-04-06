from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.layers import Dense, Input, Embedding, Concatenate
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dropout, GlobalMaxPool1D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from lib.util.preprocessing import to_extended_categorical
from lib.embedding import word2vec, fasttext
from tinydb import TinyDB
import tensorflow as tf
import numpy as np


def train(train_x, train_aux, train_y, evaluate_x, evaluate_aux, evaluate_y, embedding_map, embedding_dim, max_sequence_len, num_classes):
    with tf.device('/gpu:1'):
        sequence_input = Input(shape=(max_sequence_len,), dtype='int32', name='sequence_input')
        embedding_layer_1 = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_map],
                                      input_length=max_sequence_len, trainable=False)
        embedded_sequences_1 = embedding_layer_1(sequence_input)
        l_conv1= Conv1D(250, 10, activation='relu', padding='valid',)(embedded_sequences_1)
        l_pool1 = MaxPooling1D(5)(l_conv1)
        l_conv2 = Conv1D(150, 5, activation='relu')(l_pool1)
        l_pool3 = GlobalMaxPool1D()(l_conv2)
        l_dense1 = Dense(100, activation='relu')(l_pool3)
        l_dropout1 = Dropout(0.4)(l_dense1)

        auxiliary_input = Input(shape=(3*num_classes,), name='auxiliary_input')
        l_dense2 = Dense(10, activation='relu')(auxiliary_input)
        # l_dropout2 = Dropout(0.4)(l_dense2)

        l_concat1 = Concatenate()([l_dropout1, l_dense2])
        l_dense3 = Dense(60, activation='relu')(l_concat1)
        l_dropout3 = Dropout(0.4)(l_dense3)
        l_dense4 = Dense(20, activation='relu')(l_dropout3)
        preds = Dense(num_classes, activation='softmax')(l_dense4)

        cnn_model = Model(inputs=[sequence_input, auxiliary_input], outputs=[preds])
        early_stopping_callback = EarlyStopping(patience=2, monitor='val_acc')
        cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        cnn_model.summary()
        cnn_model.fit({"sequence_input":train_x, "auxiliary_input":train_aux}, train_y,
                      validation_data=({"sequence_input":evaluate_x, "auxiliary_input": evaluate_aux}, evaluate_y),
                      epochs=12, batch_size=64, callbacks=[early_stopping_callback])
    return cnn_model


def cross_val(data_x, data_aux, data_y, embedding_map, embedding_dim, max_sequence_len, num_classes, n_splits=5):
    skf = StratifiedKFold(n_splits)
    print("Performing cross validation (%d-fold)..." % n_splits)
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    mean_accuracy = 0
    for train_index, test_index in skf.split(data_x, data_y.argmax(axis=1)):
        cnn_pipeline = train(data_x[train_index], data_aux[train_index], data_y[train_index],
                             data_x[test_index], data_aux[test_index], data_y[test_index],
                             embedding_map, embedding_dim, max_sequence_len, num_classes)
        # metrics = evaluate(cnn_pipeline, data_x[test_index], data_y[test_index])
        # precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
        # recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
        # mean_accuracy += metrics['micro-average'][0]
        # print("Precision, Recall, F_Score, Support")
        # print(metrics)
    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/n_splits, [precision/n_splits for precision in precision_list], [recall/n_splits for recall in recall_list]))


def predict(classifier, predict_x):
    return classifier.predict(predict_x)


def evaluate(classifier, evaluate_x, evaluate_y):
    predict_y = predict(classifier, evaluate_x).argmax(axis=1)
    evaluate_y = evaluate_y.argmax(axis=1)
    return {"individual": precision_recall_fscore_support(evaluate_y, predict_y),
            "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}


if __name__ == '__main__':
    embedding_dim_1 = embedding_dim_2 = 300
    num_classes = 4
    dataset_list = ['data/labelled/pull_requests/spring.json', 'data/labelled/pull_requests/oni.json', 'data/labelled/pull_requests/tensorflow.json']
    label_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    data_x = list()
    data_y = list()
    data_aux1 = list()
    data_aux2 = list()
    data_aux3 = list()
    for dataset_path in dataset_list:
        db = TinyDB(dataset_path)
        for entry in db:
            for thread in entry.values():
                prev1_label = -1
                prev2_label = -1
                prev3_label = -1
                for comment in thread['comments']:
                    if 'ipa' in comment and (comment['ipa'] != '-'):
                        comment_body = comment['body'].lower()
                        current_label = label_map[comment['ipa'].lower()]
                        data_x.append(comment_body)
                        data_aux1.append(prev1_label)
                        data_aux2.append(prev2_label)
                        data_aux3.append(prev3_label)
                        data_y.append(current_label)
                        prev3_label = prev2_label
                        prev2_label = prev1_label
                        prev1_label = current_label

    data_x = np.array(data_x)
    print("Dataset loaded to memory. Size:", len(data_y))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data_x)
    sequences = tokenizer.texts_to_sequences(data_x)
    seq_lengths = [len(seq) for seq in sequences]
    max_sequence_len = max(seq_lengths)
    # avg_sequence_len = sum(seq_lengths)/len(seq_lengths)
    # print(max_sequence_len, avg_sequence_len)
    max_sequence_len = min(400, max_sequence_len)
    data_x = pad_sequences(sequences, maxlen=max_sequence_len)
    data_y = to_categorical(data_y, num_classes=num_classes)
    data_aux1 = to_extended_categorical(data_aux1, num_classes=num_classes)
    data_aux2 = to_extended_categorical(data_aux2, num_classes=num_classes)
    data_aux3 = to_extended_categorical(data_aux3, num_classes=num_classes)
    data_aux = np.hstack((data_aux1, data_aux2, data_aux3))
    word_index = tokenizer.word_index
    # embedding_map_1 = word2vec.embedding_matrix(word_index, model_path="data/embedding/word2vec/googlenews_size300.bin", binary=True)
    embedding_map_2 = word2vec.embedding_matrix(word_index)
    cross_val(data_x, data_aux, data_y, embedding_map_2, embedding_dim_2, max_sequence_len, num_classes, n_splits=10)