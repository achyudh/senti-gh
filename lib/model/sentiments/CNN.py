from tensorflow.python.keras.layers import Dense, Input, Embedding, Concatenate, BatchNormalization
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dropout, GlobalMaxPool1D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.models import Model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from lib.embedding import word2vec, fasttext
from sklearn.utils import resample
from lib.util import preprocessing
import tensorflow as tf
import pandas as pd


def train(train_x, train_y, evaluate_x, evaluate_y, embedding_map, embedding_dim, tokenizer, max_sequence_len, num_classes, dataset_name):
    with tf.device('/gpu:0'):
        embedding_layer_1 = Embedding(len(tokenizer.word_index) + 1, embedding_dim, weights=[embedding_map],
                                input_length=max_sequence_len, trainable=False)
        sequence_input = Input(shape=(max_sequence_len,), dtype='int32')
        embedded_sequences_1 = embedding_layer_1(sequence_input)
        l_conv1 = Conv1D(150, 5, activation='relu', padding='valid')(embedded_sequences_1)
        l_pool1 = MaxPooling1D(5)(l_conv1)
        l_conv2 = Conv1D(150, 3, activation='relu')(l_pool1)
        l_pool2 = GlobalMaxPool1D()(l_conv2)
        l_dense1 = Dense(75, activation='relu')(l_pool2)
        l_dropout1 = Dropout(0.2)(l_dense1)
        l_dense2 = Dense(50, activation='relu')(l_dropout1)
        l_dropout2 = Dropout(0.2)(l_dense2)
        preds = Dense(num_classes, activation='softmax')(l_dropout2)

        cnn_model = Model(sequence_input, preds)
        early_stopping_callback = EarlyStopping(patience=5, monitor='val_acc')
        checkpoint_callback = ModelCheckpoint(filepath="data/models/cnn/%s.hdf5" % dataset_name, monitor='val_acc', verbose=1, save_best_only=True)
        cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        cnn_model.summary()
        # plot_model(cnn_model, to_file='model.png')
        cnn_model.fit(train_x, train_y, validation_data=(evaluate_x, evaluate_y), epochs=20, batch_size=64,
                      callbacks=[early_stopping_callback, checkpoint_callback])
        cnn_model.load_weights("data/models/cnn/%s.hdf5" % dataset_name)
    return cnn_model


def predict(classifier, predict_x):
    return classifier.predict(predict_x)


def evaluate(classifier, evaluate_x, evaluate_y):
    predict_y = predict(classifier, evaluate_x).argmax(axis=1)
    evaluate_y = evaluate_y.argmax(axis=1)
    # print(accuracy_score(evaluate_y, predict_y))
    return {"individual": precision_recall_fscore_support(evaluate_y, predict_y),
            "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}


def cross_val(data_x, data_y, embedding_map, embedding_dim, max_sequence_len, num_classes, n_splits=5):
    skf = StratifiedKFold(n_splits, random_state=157)
    print("Performing cross validation (%d-fold)..." % n_splits)
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    mean_accuracy = 0
    iteration = 1
    for train_index, test_index in skf.split(data_x, data_y.argmax(axis=1)):
        print("Iteration %d of %d" % (iteration, n_splits))
        iteration += 1
        cnn_pipeline = train(data_x[train_index], data_y[train_index], data_x[test_index], data_y[test_index], embedding_map, embedding_dim, tokenizer, max_sequence_len, num_classes, dataset_name)
        metrics = evaluate(cnn_pipeline, data_x[test_index], data_y[test_index])
        precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
        recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
        mean_accuracy += metrics['micro-average'][0]
        print("Precision, Recall, F_Score, Support")
        print(metrics)
    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/n_splits, [precision/n_splits for precision in precision_list], [recall/n_splits for recall in recall_list]))


def bootstrap_trend(data_list, embedding_dim, num_classes):
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    accuracy_list = list()

    for i0 in range(len(data_list)):
        data_list[i0] = resample(data_list[i0], n_samples=1500, random_state=157, replace=False)
    data = pd.concat(data_list).as_matrix()
    data_x, data_y_cat, tokenizer, max_sequence_len = preprocessing.make_network_ready(data, num_classes)
    embedding_map = word2vec.embedding_matrix(tokenizer.word_index, model_path="data/embedding/word2vec/googlenews_size300.bin", binary=True)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y_cat, test_size=0.3, random_state=157, stratify=data_y_cat)
    print("Metrics: Precision, Recall, F_Score, Support")

    for sample_rate in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if sample_rate == 1.0:
            train_xr, train_yr = train_x, train_y
        else:
            n_samples = int(sample_rate * len(train_y) + 1)
            train_xr, train_yr = resample(train_x, train_y, n_samples=n_samples, random_state=157)
        cnn_pipeline = train(train_xr, train_yr, test_x, test_y, embedding_map, embedding_dim, tokenizer, max_sequence_len, num_classes, "Combined_%f" % sample_rate)
        metrics = evaluate(cnn_pipeline, test_x, test_y)
        print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))
        precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
        recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
        accuracy_list.append(metrics['micro-average'][0])
    print(accuracy_list)
    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (sum(accuracy_list)/9, [precision/9 for precision in precision_list], [recall/9 for recall in recall_list]))


def hard_cross_val(data_list, embedding_map, embedding_dim, tokenizer, max_sequence_len, num_classes):
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    mean_accuracy = 0

    for i0 in range(len(data_list)):
        data_list[i0] = resample(data_list[i0], n_samples=1500, random_state=157, replace=False)

    for i0 in range(len(data_list)):
        data_test = data_list[i0].as_matrix()
        data_train = list()
        for i1 in range(len(data_list)):
            if i1 != i0:
                data_train.append(data_list[i1])
        data_train = pd.concat(data_train).as_matrix()
        train_x, train_y_cat, _word_index, _max_sequence_len = preprocessing.make_network_ready(data_train, num_classes, tokenizer, max_sequence_len, enforce_max_len=True)
        test_x, test_y_cat, _word_index, _max_sequence_len = preprocessing.make_network_ready(data_test, num_classes, tokenizer, max_sequence_len, enforce_max_len=True)
        cnn_pipeline = train(train_x, train_y_cat, test_x, test_y_cat, embedding_map, embedding_dim, max_sequence_len, tokenizer, num_classes, dataset_name)
        metrics = evaluate(cnn_pipeline, test_x, test_y_cat)
        mean_accuracy += metrics['micro-average'][0]
        precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
        recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
        print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))

    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/len(data_list), [precision/len(data_list) for precision in precision_list],
                                                                     [recall/len(data_list) for recall in recall_list]))


if __name__ == '__main__':
    num_classes = 3
    embedding_dim = 300
    dataset_name = 'Combined'
    # data = pd.read_csv("data/labelled/Gerrit.csv").as_matrix()
    # data = pd.read_csv("data/labelled/StackOverflowJavaLibraries.csv", encoding='latin1').as_matrix()
    # data_1 = pd.read_csv("data/labelled/Gerrit.csv")
    # data_2 = pd.read_csv("data/labelled/JIRA.csv")
    # data_3 = pd.read_csv("data/labelled/AppReviews2.csv")
    data_4 = pd.read_csv("data/labelled/StackOverflowEmotions.csv", encoding='latin1')
    data_5 = pd.read_csv("data/labelled/StackOverflowSentiments.csv", encoding='latin1')
    data_6 = pd.read_csv("data/labelled/StackOverflowJavaLibraries.csv", encoding='latin1')
    data_list = [data_4, data_5, data_6]
    # data = pd.concat(data_list).as_matrix()
    # data_x, data_y_cat, tokenizer, max_sequence_len = preprocessing.make_network_ready(data, num_classes)
    # print("Dataset loaded to memory. Size:", len(data_y_cat))
    # embedding_map = word2vec.embedding_matrix(tokenizer.word_index, model_path="data/embedding/word2vec/googlenews_size300.bin", binary=True)
    # embedding_map = word2vec.embedding_matrix(tokenizer.word_index)
    # cross_val(data_x, data_y_cat, embedding_map, embedding_dim, max_sequence_len, num_classes, n_splits=10)
    bootstrap_trend(data_list, embedding_dim, num_classes)
    # hard_cross_val(data_list, embedding_map, embedding_dim, max_sequence_len, tokenizer, num_classes)