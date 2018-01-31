from nltk.tokenize import word_tokenize
from tinydb import TinyDB
import pandas as pd
import numpy as np
import os


def labelled_comments(dataset_path, delete_identifier=True):
    raw_data = pd.read_csv(dataset_path)
    if delete_identifier and 'pull_request_id' in raw_data.columns:
        del raw_data['pull_request_id']
    raw_data = raw_data.as_matrix()
    np.random.shuffle(raw_data)
    data_y = raw_data[:, 1].astype('int')
    raw_x = raw_data[:, 0]
    data_x = list()
    for sentence in raw_x:
        if isinstance(sentence, str):
            data_x.append(word_tokenize(sentence.lower()))
        else:
            data_x.append([""])
    data_x = np.array(data_x)
    return data_x, data_y


def complete_text(dataset_path="./data/user/"):
    token_matrix = list()
    for subdir, dirs, files in os.walk(dataset_path):
        print("Processing %s..." % subdir)
        for file in files:
            db = TinyDB(os.path.join(subdir, file))
            for entry in db:
                for user_data in entry.values():
                    for value in user_data['issues']:
                        if value['title'] is not None and value['body'] is not None:
                            token_matrix.append(word_tokenize((value['title'] + ' ' + value['body']).lower()))
                        elif value['body'] is not None:
                            token_matrix.append(word_tokenize(value['body'].lower()))
                        elif value['title'] is not None:
                            token_matrix.append(word_tokenize(value['title'].lower()))

                    for value in user_data['issue_comments']:
                        if value['body'] is not None:
                            token_matrix.append(word_tokenize(value['body'].lower()))

                    for value in user_data['pull_requests']:
                        if value['title'] is not None and value['body'] is not None:
                            token_matrix.append(word_tokenize((value['title'] + ' ' + value['body']).lower()))
                        elif value['body'] is not None:
                            token_matrix.append(word_tokenize(value['body'].lower()))
                        elif value['title'] is not None:
                            token_matrix.append(word_tokenize(value['title'].lower()))

                    for value in user_data['review_comments']:
                        if value['body'] is not None:
                            token_matrix.append(word_tokenize(value['body'].lower()))

                    for value in user_data['commits']:
                        if value['message'] is not None:
                            token_matrix.append(word_tokenize(value['message'].lower()))

                    for value in user_data['commit_comments']:
                        if value['body'] is not None:
                            token_matrix.append(word_tokenize(value['body'].lower()))
    return np.array(token_matrix)


def review_comments():
    pass


if __name__ == '__main__':
    print(labelled_comments("./data/labelled/pull_requests/grouped_emotions.csv"))