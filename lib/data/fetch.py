from nltk.tokenize import word_tokenize
from tinydb import TinyDB
import pandas as pd
import numpy as np
import os, random


def labelled_comments(dataset_path, tokenize=True, delete_identifier=True):
    raw_data = pd.read_csv(dataset_path)
    if delete_identifier and 'pull_request_id' in raw_data.columns:
        del raw_data['pull_request_id']
    raw_data = raw_data.as_matrix()
    np.random.shuffle(raw_data)
    data_y = raw_data[:, 1].astype('int')
    raw_x = raw_data[:, 0]
    data_x = list()
    if tokenize:
        for sentence in raw_x:
            if isinstance(sentence, str):
                data_x.append(word_tokenize(sentence.lower()))
            else:
                data_x.append([""])
    else:
        for sentence in raw_x:
            if isinstance(sentence, str):
                data_x.append(sentence.lower())
            else:
                data_x.append("")
    data_x = np.array(data_x)
    return data_x, data_y


def complete_text(dataset_path="./data/user/", tokenize=True):
    token_matrix = list()
    if tokenize:
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
    else:
        for subdir, dirs, files in os.walk(dataset_path):
            print("Processing %s..." % subdir)
            for file in files:
                db = TinyDB(os.path.join(subdir, file))
                for entry in db:
                    for user_data in entry.values():
                        for value in user_data['issues']:
                            if value['title'] is not None and value['body'] is not None:
                                token_matrix.append((value['title'] + ' ' + value['body']).lower())
                            elif value['body'] is not None:
                                token_matrix.append(value['body'].lower())
                            elif value['title'] is not None:
                                token_matrix.append(value['title'].lower())

                        for value in user_data['issue_comments']:
                            if value['body'] is not None:
                                token_matrix.append(value['body'].lower())

                        for value in user_data['pull_requests']:
                            if value['title'] is not None and value['body'] is not None:
                                token_matrix.append((value['title'] + ' ' + value['body']).lower())
                            elif value['body'] is not None:
                                token_matrix.append(value['body'].lower())
                            elif value['title'] is not None:
                                token_matrix.append(value['title'].lower())

                        for value in user_data['review_comments']:
                            if value['body'] is not None:
                                token_matrix.append(value['body'].lower())

                        for value in user_data['commits']:
                            if value['message'] is not None:
                                token_matrix.append(value['message'].lower())

                        for value in user_data['commit_comments']:
                            if value['body'] is not None:
                                token_matrix.append(value['body'].lower())
    print("Collected %d comments in total" % len(token_matrix))
    return np.array(token_matrix)


def text_with_reactions(rootdir, tokenize=True):
    total_count = reaction_count = no_reaction_count = 0
    token_matrix = list()
    reaction_matrix = list()
    if tokenize:
        for subdir, dirs, files in os.walk(rootdir):
            print("Processing %s..." % subdir)
            for file in files:
                db = TinyDB(os.path.join(subdir, file))
                for entry in db:
                    for user_data in entry.values():
                        total_count += 1
                        for issue in user_data['issues']:
                            if issue['reactions']['total_count'] > 0:
                                reaction_count += 1
                                null_flag = False
                                if issue['title'] is not None and issue['body'] is not None:
                                    token_matrix.append(word_tokenize((issue['title'] + ' ' + issue['body']).lower()))
                                elif issue['body'] is not None:
                                    token_matrix.append(word_tokenize(issue['body'].lower()))
                                elif issue['title'] is not None:
                                    token_matrix.append(word_tokenize(issue['title'].lower()))
                                else:
                                    null_flag = True
                                if not null_flag:
                                    reaction_matrix.append((1 if issue['reactions']['+1'] > 0 else 0,
                                                          1 if issue['reactions']['-1'] > 0 else 0,
                                                          1 if issue['reactions']['laugh'] > 0 else 0,
                                                          1 if issue['reactions']['hooray'] > 0 else 0,
                                                          1 if issue['reactions']['confused'] > 0 else 0,
                                                          1 if issue['reactions']['heart'] > 0 else 0))
                        else:
                            if random.randint(0, 100) < 2:
                                no_reaction_count += 1
                                null_flag = False
                                if issue['title'] is not None and issue['body'] is not None:
                                    token_matrix.append(word_tokenize((issue['title'] + ' ' + issue['body']).lower()))
                                elif issue['body'] is not None:
                                    token_matrix.append(word_tokenize(issue['body'].lower()))
                                elif issue['title'] is not None:
                                    token_matrix.append(word_tokenize(issue['title'].lower()))
                                else:
                                    null_flag = True
                                if not null_flag:
                                    reaction_matrix.append((0, 0, 0, 0, 0, 0))

                        for comment in user_data['issue_comments']:
                            if comment['reactions']['total_count'] > 0:
                                reaction_count += 1
                                if comment['body'] is not None:
                                    token_matrix.append(word_tokenize(comment['body'].lower()))
                                    reaction_matrix.append((1 if comment['reactions']['+1'] > 0 else 0,
                                                           1 if comment['reactions']['-1'] > 0 else 0,
                                                           1 if comment['reactions']['laugh'] > 0 else 0,
                                                           1 if comment['reactions']['hooray'] > 0 else 0,
                                                           1 if comment['reactions']['confused'] > 0 else 0,
                                                           1 if comment['reactions']['heart'] > 0 else 0))
                            else:
                                if random.randint(0, 100) < 2:
                                    no_reaction_count += 1
                                    if comment['body'] is not None:
                                        token_matrix.append(word_tokenize(comment['body'].lower()))
                                        reaction_matrix.append((0, 0, 0, 0, 0, 0))

                        for comment in user_data['review_comments']:
                            if comment['reactions']['total_count'] > 0:
                                reaction_count += 1
                                if comment['body'] is not None:
                                    token_matrix.append(word_tokenize(comment['body'].lower()))
                                    reaction_matrix.append((1 if comment['reactions']['+1'] > 0 else 0,
                                                            1 if comment['reactions']['-1'] > 0 else 0,
                                                            1 if comment['reactions']['laugh'] > 0 else 0,
                                                            1 if comment['reactions']['hooray'] > 0 else 0,
                                                            1 if comment['reactions']['confused'] > 0 else 0,
                                                            1 if comment['reactions']['heart'] > 0 else 0))
                            else:
                                if random.randint(0, 100) < 2:
                                    no_reaction_count += 1
                                    if comment['body'] is not None:
                                        token_matrix.append(word_tokenize(comment['body'].lower()))
                                        reaction_matrix.append((0, 0, 0, 0, 0, 0))

                        for comment in user_data['commit_comments']:
                            if comment['reactions']['total_count'] > 0:
                                reaction_count += 1
                                if comment['body'] is not None:
                                    token_matrix.append(word_tokenize(comment['body'].lower()))
                                    reaction_matrix.append((1 if comment['reactions']['+1'] > 0 else 0,
                                                            1 if comment['reactions']['-1'] > 0 else 0,
                                                            1 if comment['reactions']['laugh'] > 0 else 0,
                                                            1 if comment['reactions']['hooray'] > 0 else 0,
                                                            1 if comment['reactions']['confused'] > 0 else 0,
                                                            1 if comment['reactions']['heart'] > 0 else 0))
                            else:
                                if random.randint(0, 100) < 2:
                                    no_reaction_count += 1
                                    if comment['body'] is not None:
                                        token_matrix.append(word_tokenize(comment['body'].lower()))
                                        reaction_matrix.append((0, 0, 0, 0, 0, 0))
    else:
        for subdir, dirs, files in os.walk(rootdir):
            print("Processing %s..." % subdir)
            for file in files:
                db = TinyDB(os.path.join(subdir, file))
                for entry in db:
                    for user_data in entry.values():
                        total_count += 1
                        for issue in user_data['issues']:
                            if issue['reactions']['total_count'] > 0:
                                reaction_count += 1
                                null_flag = False
                                if issue['title'] is not None and issue['body'] is not None:
                                    token_matrix.append((issue['title'] + ' ' + issue['body']).lower())
                                elif issue['body'] is not None:
                                    token_matrix.append(issue['body'].lower())
                                elif issue['title'] is not None:
                                    token_matrix.append(issue['title'].lower())
                                else:
                                    null_flag = True
                                if not null_flag:
                                    reaction_matrix.append((1 if issue['reactions']['+1'] > 0 else 0,
                                                            1 if issue['reactions']['-1'] > 0 else 0,
                                                            1 if issue['reactions']['laugh'] > 0 else 0,
                                                            1 if issue['reactions']['hooray'] > 0 else 0,
                                                            1 if issue['reactions']['confused'] > 0 else 0,
                                                            1 if issue['reactions']['heart'] > 0 else 0))
                        else:
                            if random.randint(0, 100) < 2:
                                no_reaction_count += 1
                                null_flag = False
                                if issue['title'] is not None and issue['body'] is not None:
                                    token_matrix.append((issue['title'] + ' ' + issue['body']).lower())
                                elif issue['body'] is not None:
                                    token_matrix.append(issue['body'].lower())
                                elif issue['title'] is not None:
                                    token_matrix.append(issue['title'].lower())
                                else:
                                    null_flag = True
                                if not null_flag:
                                    reaction_matrix.append((0, 0, 0, 0, 0, 0))

                        for comment in user_data['issue_comments']:
                            if comment['reactions']['total_count'] > 0:
                                reaction_count += 1
                                if comment['body'] is not None:
                                    token_matrix.append(comment['body'].lower())
                                    reaction_matrix.append((1 if comment['reactions']['+1'] > 0 else 0,
                                                            1 if comment['reactions']['-1'] > 0 else 0,
                                                            1 if comment['reactions']['laugh'] > 0 else 0,
                                                            1 if comment['reactions']['hooray'] > 0 else 0,
                                                            1 if comment['reactions']['confused'] > 0 else 0,
                                                            1 if comment['reactions']['heart'] > 0 else 0))
                            else:
                                if random.randint(0, 100) < 2:
                                    no_reaction_count += 1
                                    if comment['body'] is not None:
                                        token_matrix.append(comment['body'].lower())
                                        reaction_matrix.append((0, 0, 0, 0, 0, 0))

                        for comment in user_data['review_comments']:
                            if comment['reactions']['total_count'] > 0:
                                reaction_count += 1
                                if comment['body'] is not None:
                                    token_matrix.append(comment['body'].lower())
                                    reaction_matrix.append((1 if comment['reactions']['+1'] > 0 else 0,
                                                            1 if comment['reactions']['-1'] > 0 else 0,
                                                            1 if comment['reactions']['laugh'] > 0 else 0,
                                                            1 if comment['reactions']['hooray'] > 0 else 0,
                                                            1 if comment['reactions']['confused'] > 0 else 0,
                                                            1 if comment['reactions']['heart'] > 0 else 0))
                            else:
                                if random.randint(0, 100) < 2:
                                    no_reaction_count += 1
                                    if comment['body'] is not None:
                                        token_matrix.append(comment['body'].lower())
                                        reaction_matrix.append((0, 0, 0, 0, 0, 0))

                        for comment in user_data['commit_comments']:
                            if comment['reactions']['total_count'] > 0:
                                reaction_count += 1
                                if comment['body'] is not None:
                                    token_matrix.append(comment['body'].lower())
                                    reaction_matrix.append((1 if comment['reactions']['+1'] > 0 else 0,
                                                            1 if comment['reactions']['-1'] > 0 else 0,
                                                            1 if comment['reactions']['laugh'] > 0 else 0,
                                                            1 if comment['reactions']['hooray'] > 0 else 0,
                                                            1 if comment['reactions']['confused'] > 0 else 0,
                                                            1 if comment['reactions']['heart'] > 0 else 0))
                            else:
                                if random.randint(0, 100) < 2:
                                    no_reaction_count += 1
                                    if comment['body'] is not None:
                                        token_matrix.append(comment['body'].lower())
                                        reaction_matrix.append((0, 0, 0, 0, 0, 0))


    print("Number of comments with reactions:", reaction_count, "Number of added comments without reactions:", reaction_count, "Total number of comments:", total_count, "Fraction:", (reaction_count+no_reaction_count)/total_count)
    return np.array(token_matrix), np.array(reaction_matrix).astype('int')



if __name__ == '__main__':
    print(labelled_comments("./data/labelled/pull_requests/grouped_emotions.csv"))