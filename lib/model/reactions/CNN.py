from lib.embedding.word2vec import embedding_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.layers import Embedding, Dense, Input, Flatten, Embedding
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from lib.data import fetch
import numpy as np


def train(train_x, train_y, evaluate_x, evaluate_y, embedding_map, embedding_dim, max_sequence_len):
    embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_map],
                            input_length=max_sequence_len, trainable=False)
    sequence_input = Input(shape=(max_sequence_len,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
    l_pool1 = MaxPooling1D(4)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(4)(l_cov2)
    l_cov3 = Conv1D(64, 2, activation='relu')(l_pool2)
    l_pool3 = MaxPooling1D(16)(l_cov3)
    l_flat = Flatten()(l_pool3)
    l_dense1 = Dense(128, activation='sigmoid')(l_flat)
    # l_dropout = Dropout(0.2)(l_dense1)
    l_dense2 = Dense(64, activation='sigmoid')(l_dense1)
    preds = Dense(1, activation='sigmoid')(l_dense2)
    cnn_model = Model(sequence_input, preds)
    cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnn_model.summary()
    cnn_model.fit(train_x, train_y, validation_data=(evaluate_x, evaluate_y), epochs=10, batch_size=256)
    return cnn_model


def predict(classifier, predict_x):
    return classifier.predict(predict_x)


def evaluate(classifier, evaluate_x, evaluate_y):
    predict_y = predict(classifier, evaluate_x)
    return {"individual":precision_recall_fscore_support(evaluate_y, predict_y),
            "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}


def cross_val(data_x, data_y, embedding_map, embedding_dim, max_sequence_len, n_splits=5):
    skf = StratifiedKFold(n_splits)
    print("Performing cross validation (%d-fold)..." % n_splits)
    mean_precision = 0
    mean_recall = 0
    mean_accuracy = 0
    for train_index, test_index in skf.split(data_x, data_y):
        cnn_pipeline = train(data_x[train_index], data_y[train_index], data_x[test_index], data_y[test_index], embedding_map, embedding_dim, max_sequence_len)
        metrics = evaluate(cnn_pipeline, data_x[test_index], data_y[test_index])
        mean_precision += metrics['individual'][0][1]
        mean_recall += metrics['individual'][1][1]
        mean_accuracy += metrics['micro-average'][0]
        print("Precision, Recall, F_Score, Support")
        print(metrics)
    print("Mean precision: %s, Mean recall: %s Mean accuracy: %s" % (mean_precision/n_splits, mean_recall/n_splits, mean_accuracy/n_splits))


if __name__ == '__main__':
    embedding_dim = 100
    data_x, reaction_matrix = fetch.sentences_with_reactions("data/user", tokenize=False)
    data_y = reaction_matrix[:, 0]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data_x)
    sequences = tokenizer.texts_to_sequences(data_x)
    max_sequence_len = max(len(seq) for seq in sequences)
    max_sequence_len = min(500, max_sequence_len)
    print("Max. sequence length:", max_sequence_len)
    data_x = pad_sequences(sequences, maxlen=max_sequence_len)
    # data_y_cat = to_categorical(data_y, num_classes=2)
    word_index = tokenizer.word_index
    embedding_map = embedding_matrix(word_index, model_path='data/embedding/word2vec/gensim_size100_min3')
    print("Thumbs-up:")
    cross_val(data_x, data_y, embedding_map, embedding_dim, max_sequence_len, n_splits=5)