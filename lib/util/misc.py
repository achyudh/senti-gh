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


if __name__ == '__main__':
    remove_duplicate_entries("data/user/tensorflow/tensorflow.json")
