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


def plot_line_graph(arrays, x_label="Percentage of training set used", y_label="Accuracy", figsize=(8, 6), dpi=100,
                    ylim=(0.75, 1), filename="bootstrap_trend.png", xticks=(20, 30, 40, 50, 60, 70, 80, 90, 100)):
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


if __name__ == '__main__':
    plot_line_graph([
        {"data": [0.8211610486891385, 0.8314606741573034, 0.8314606741573034, 0.8417602996254682, 0.8389513108614233, 0.8548689138576779, 0.8604868913857678, 0.8539325842696629, 0.849250936329588],
         "label": "CNN"},
        {"data":[0.7584269662921348, 0.7762172284644194, 0.7865168539325843, 0.7846441947565543, 0.8089887640449438, 0.799625468164794, 0.8061797752808989, 0.8239700374531835, 0.8192883895131086],
         "label": "Naive Bayes (NLTK)"},
        {"data": [0.8332396477421773, 0.8332396477421773, 0.8332396477421773, 0.8332396477421773, 0.8332396477421773, 0.8332396477421773, 0.8332396477421773, 0.8332396477421773, 0.8332396477421773],
        "label": "SentiStrength SE"},
        {"data":  [0.8202247191011236, 0.8080524344569289, 0.8295880149812734, 0.8314606741573034, 0.8417602996254682, 0.8426966292134831, 0.8398876404494382, 0.8398876404494382, 0.8408239700374532],
         "label": "SentiCR"}
    ])
    # Unsupervised classifier accuracy trends:
    # {"data": [0.8436329588014981, 0.8436329588014981, 0.8436329588014981, 0.8436329588014981, 0.8436329588014981, 0.8436329588014981, 0.8436329588014981, 0.8436329588014981, 0.8436329588014981],
    #  "label": "VADER (NLTK)"},
    # {"data": [0.8444819186809068, 0.8444819186809068, 0.8444819186809068, 0.8444819186809068, 0.8444819186809068, 0.8444819186809068, 0.8444819186809068, 0.8444819186809068, 0.8444819186809068],
    #  "label": "SentiStrength"},
