import os
import time

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Conv1D, Dropout, GlobalMaxPool1D, Concatenate
from tensorflow.python.keras.layers import Dense, Input, Embedding, LSTM, Bidirectional, TimeDistributed
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam


def initialize(embedding_map, tokenizer, config):
    embedding_layer_1 = Embedding(len(tokenizer.word_index) + 1, config.embedding_dim,
                                  weights=[embedding_map],
                                  input_length=config.max_sequence_len,
                                  trainable=False)
    sequence_input_1 = Input(shape=(config.max_sequence_len,), dtype='int32')
    embedded_sequences_1 = embedding_layer_1(sequence_input_1)
    l_conv1 = Conv1D(100, 5, activation='relu', padding='valid')(embedded_sequences_1)
    l_pool1 = GlobalMaxPool1D()(l_conv1)
    l_conv2 = Conv1D(100, 4, activation='relu', padding='valid')(embedded_sequences_1)
    l_pool2 = GlobalMaxPool1D()(l_conv2)
    l_conv3 = Conv1D(100, 3, activation='relu', padding='valid')(embedded_sequences_1)
    l_pool3 = GlobalMaxPool1D()(l_conv3)
    l_concat1 = Concatenate()([l_pool1, l_pool2, l_pool3])
    l_dense1 = Dense(config.bottleneck_dim, activation='relu')(l_concat1)
    l_dropout1 = Dropout(config.dropout_rate)(l_dense1)
    encoder_1 = Model(sequence_input_1, l_dropout1)

    sequence_input_2 = Input(shape=(config.max_sequences, config.max_sequence_len), dtype='int32')
    encoder_2 = TimeDistributed(encoder_1)(sequence_input_2)
    l_lstm_2 = Bidirectional(LSTM(config.hidden_dim, dropout=0.2, recurrent_dropout=0.2))(encoder_2)
    preds = Dense(config.num_classes, activation='softmax')(l_lstm_2)
    model = Model(sequence_input_2, preds)
    optim = Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    model.summary()
    return model


def train(model, train_x, train_y, evaluate_x, evaluate_y, config):
    early_stopping_callback = EarlyStopping(patience=config.patience, monitor='val_acc')
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(config.checkpoint_path, "HiCNNBiLSTM-%s.hdf5" % config.dataset),
                                          monitor='val_acc',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)
    model.fit(train_x, train_y,
              validation_data=(evaluate_x, evaluate_y),
              epochs=config.epochs,
              batch_size=config.batch_size,
              callbacks=[early_stopping_callback, checkpoint_callback])
    model.load_weights(os.path.join(config.checkpoint_path, "HiCNNBiLSTM-%s.hdf5" % config.dataset))
    return model


def predict(model, predict_x):
    return model.predict(predict_x)


def evaluate(model, evaluate_x, evaluate_y):
    predict_y = predict(model, evaluate_x).argmax(axis=1)
    evaluate_y = evaluate_y.argmax(axis=1)
    return {"individual": precision_recall_fscore_support(evaluate_y, predict_y),
            "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}


def cross_val(data_x, data_y, embedding_map, tokenizer, config, n_splits=5):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=157)
    print("Performing cross validation (%d-fold)..." % n_splits)
    precision_list = [0 for i in range(config.num_classes)]
    recall_list = [0 for i in range(config.num_classes)]
    f1_list = [0 for i in range(config.num_classes)]
    mean_accuracy = 0
    iteration = 1
    for train_index, test_index in skf.split(data_x, data_y.argmax(axis=1)):
        print("Iteration %d of %d" % (iteration, n_splits))
        iteration += 1

        model = initialize(embedding_map, tokenizer, config)
        if config.load_model:
            model.load_weights(os.path.join(config.load_path))

        train_start_time = time.time()
        model = train(model, data_x[train_index], data_y[train_index], data_x[test_index], data_y[test_index], config)
        print("Train time:", time.time() - train_start_time)

        test_start_time = time.time()
        metrics = evaluate(model, data_x[test_index], data_y[test_index])
        print("Test time:", time.time() - test_start_time)

        precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
        recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
        f1_list = [x + y for x, y in zip(metrics['individual'][2], f1_list)]
        print("Accuracy: %s, Precision: %s, Recall: %s, F1: %s" % (metrics['micro-average'][0],
                                                                   metrics['individual'][0],
                                                                   metrics['individual'][1],
                                                                   metrics['individual'][2]))
        mean_accuracy += metrics['micro-average'][0]
    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s, Mean F1: %s" % (mean_accuracy/n_splits,
                                                                                  [precision/n_splits for precision in precision_list],
                                                                                  [recall/n_splits for recall in recall_list],
                                                                                  [f1/n_splits for f1 in f1_list]))


def bootstrap_trend(data_x, data_y_cat, embedding_map, tokenizer, config):
    precision_list = [0 for i in range(config.num_classes)]
    recall_list = [0 for i in range(config.num_classes)]
    f1_list = [0 for i in range(config.num_classes)]
    accuracy_list = list()
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y_cat,
                                                        test_size=0.3,
                                                        random_state=157,
                                                        stratify=data_y_cat)

    for sample_rate in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if sample_rate == 1.0:
            train_xr, train_yr = train_x, train_y
        else:
            n_samples = int(sample_rate * len(train_y) + 1)
            train_xr, train_yr = resample(train_x, train_y, n_samples=n_samples, random_state=157)
        model = initialize(embedding_map, tokenizer, config)
        cnn_pipeline = train(model, train_xr, train_yr, test_x, test_y, config)
        metrics = evaluate(cnn_pipeline, test_x, test_y)
        print("Accuracy: %s, Precision: %s, Recall: %s, F1: %s" % (metrics['micro-average'][0], metrics['individual'][0],
                                                                   metrics['individual'][1], metrics['individual'][2]))
        precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
        recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
        f1_list = [x + y for x, y in zip(metrics['individual'][2], f1_list)]
        accuracy_list.append(metrics['micro-average'][0])

    print("Accuracies:", accuracy_list)
    print("Dataset sizes:", [int(sample_rate * len(train_y) + 1) for sample_rate in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s, Mean F1: %s" % (sum(accuracy_list)/9,
                                                                                  [precision/9 for precision in precision_list],
                                                                                  [recall/9 for recall in recall_list],
                                                                                  [f1/9 for f1 in f1_list]))