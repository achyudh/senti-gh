from tensorflow.python.keras.preprocessing.text import Tokenizer
from statistics import median
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


def shuffle_list(a):
    np.random.shuffle(a)
    return a


def shuffle_lists(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)
    return a, b


def filter_repo_reaper(dataset_path):
    reaper_df = pd.read_csv(dataset_path)
    print("Repos left before filtering:", len(reaper_df))
    reaper_df[['stars', 'scorebased_org', 'randomforest_org', 'scorebased_utl', 'randomforest_utl']] = reaper_df[['stars', 'scorebased_org', 'randomforest_org', 'scorebased_utl', 'randomforest_utl']].apply(pd.to_numeric, errors='coerce')
    reaper_df = reaper_df.query('stars > 100 and scorebased_org == 1 and randomforest_org == 1 and scorebased_utl == 1 and randomforest_utl == 1')
    print("Repos left after filtering:", len(reaper_df))
    reaper_df.to_csv("reaper_100.csv", columns=['repository'])


def remove_duplicate_entries(dataset):
    with open(dataset, 'r') as json_file:
        db = json.load(json_file)
        max_entry = max([int(x) for x in db["_default"].keys()])
        for i in range(1, max_entry):
            del db["_default"][str(i)]

    with open(dataset, 'w') as json_file:
        json.dump(db, json_file)


def plot_line_graph(arrays, x_label="Number of training samples", y_label="Accuracy", figsize=(8, 6), dpi=100,
                    ylim=(0.70, 1), filename="bootstrap_trend.png", xticks=(20, 30, 40, 50, 60, 70, 80, 90, 100)):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.xticks(range(0, len(xticks)), xticks)
    plt.ylim(ylim)
    for array in arrays:
        plt.plot(array["data"], label=array["label"])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(filename)


def enumerate_file(file_path):
    file_contents = list()
    iter = 1
    with open(file_path, encoding='utf-8') as text_file:
        _temp = text_file.readline()
        for line in text_file:
            file_contents.append(str(iter) + "\t" + line)
            iter += 1
    with open(file_path, 'w', encoding='utf-8') as text_file:
        for line in file_contents:
            text_file.write(line)


def denumerate_file(file_path):
    file_contents = list()
    with open(file_path, encoding='utf-8') as text_file:
        for line in text_file:
            file_contents.append(line.split('\t', maxsplit=1)[1])
    with open(file_path, 'w', encoding='utf-8') as text_file:
        for line in file_contents:
            text_file.write(line)


def average_word_count():
    data_1 = pd.read_csv("data/labelled/JIRA.csv")
    data_2 = pd.read_csv("data/labelled/AppReviews.csv")
    data_3 = pd.read_csv("data/labelled/Gerrit.csv")
    data_4 = pd.read_csv("data/labelled/StackOverflowEmotions.csv", encoding='latin1')
    data_5 = pd.read_csv("data/labelled/StackOverflowSentiments.csv", encoding='latin1')
    data_6 = pd.read_csv("data/labelled/StackOverflowJavaLibraries.csv", encoding='latin1')
    data_list = [data_1, data_2, data_3, data_4, data_5, data_6]
    for dataset in data_list:
        data = dataset.as_matrix()
        tokenizer = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(data[:, 0])
        sequences = tokenizer.texts_to_sequences(data[:, 0])
        mean_seq_length = sum((len(seq) for seq in sequences))/len(sequences)
        median_seq_length = max((len(seq) for seq in sequences))
        print(mean_seq_length, median_seq_length)


if __name__ == '__main__':
    plot_line_graph([
        {"data":
             [0.8273381294964028, 0.8669064748201439, 0.8705035971223022, 0.9064748201438849, 0.9136690647482014, 0.9172661870503597, 0.9136690647482014, 0.9136690647482014, 0.920863309352518],
         "label": "Naive Bayes"},
        {"data":
             [0.7473118279569892, 0.7473118279569892, 0.7473118279569892, 0.7473118279569892, 0.7473118279569892,
              0.7473118279569892, 0.7473118279569892, 0.7473118279569892, 0.7473118279569892],
         "label": "VADER"},
        {"data":
             [0.9244604316546763, 0.935251798561151, 0.9316546762589928, 0.935251798561151, 0.9316546762589928,
              0.9280575539568345, 0.9316546762589928, 0.9316546762589928, 0.9244604316546763],
         "label": "Senti4SD"},
        {"data":
             [0.9028776978417267, 0.8812949640287769, 0.8884892086330936, 0.8633093525179856, 0.8776978417266187,
              0.9028776978417267, 0.9028776978417267, 0.9136690647482014, 0.9100719424460432],
         "label": "SentiCR"},
        {"data":
             [0.9532374100719424, 0.960431654676259, 0.9640287769784173, 0.9676258992805755, 0.9748201438848921,
              0.9748201438848921, 0.9784172661870504, 0.9748201438848921, 0.9748201438848921],
         "label": "Hi-CNN-LSTM"}
    ], xticks=[130, 195, 260, 325, 389, 454, 519, 584, 649], filename="jira_trend.png", ylim=(0.60, 1))
