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


def plot_line_graph(arrays, x_label="Percentage of training set used", y_label="Accuracy", ylim=(0.6, 1),
                    filename="bootstrap_trend.png", xticks=(20, 30, 40, 50, 60, 70, 80, 90, 100)):
    plt.xticks(range(0, len(xticks)), xticks)
    plt.ylim(ylim)
    for array in arrays:
        plt.plot(array["data"], label=array["label"])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(filename)


if __name__ == '__main__':
    plot_line_graph([
        {"data": [0.7855805243445693, 0.8136704119850188, 0.7734082397003745, 0.8164794007490637, 0.8323970037453183, 0.8342696629213483, 0.8623595505617978, 0.8614232209737828, 0.8576779026217228],
         "label": "CNN-LSTM"},
        {"data": [0.8071161048689138, 0.8080524344569289, 0.8099250936329588, 0.8352059925093633, 0.8370786516853933, 0.849250936329588, 0.8473782771535581, 0.8389513108614233, 0.851123595505618],
         "label": "CNN"},
        {"data":[0.7584269662921348, 0.7762172284644194, 0.7865168539325843, 0.7846441947565543, 0.8089887640449438, 0.799625468164794, 0.8061797752808989, 0.8239700374531835, 0.8192883895131086],
         "label": "Naive Bayes"}
    ])