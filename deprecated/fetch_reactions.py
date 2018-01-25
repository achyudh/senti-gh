from tinydb import TinyDB
from lib.util.preprocessing import tokenize
import numpy as np
import os


def reactions(rootdir):
    total_count = reaction_count = 0

    token_matrix = list()
    reaction_matrix = list()
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            print("Processing %s..." % os.path.join(subdir, file))
            db = TinyDB(os.path.join(subdir, file))
            for entry in db:
                for user_data in entry.values():
                    total_count += 1
                    for issue in user_data['issues']:
                        if issue['reactions']['total_count'] > 0:
                            reaction_count += 1
                            token_matrix.append(tokenize(issue['title'] + ' ' + issue['body']))
                            reaction_matrix.append((1 if issue['reactions']['+1'] > 0 else 0,
                                                  1 if issue['reactions']['-1'] > 0 else 0,
                                                  1 if issue['reactions']['laugh'] > 0 else 0,
                                                  1 if issue['reactions']['hooray'] > 0 else 0,
                                                  1 if issue['reactions']['confused'] > 0 else 0,
                                                  1 if issue['reactions']['heart'] > 0 else 0))
                        for comment in user_data['issue_comments']:
                            if comment['reactions']['total_count'] > 0:
                                reaction_count += 1
                                token_matrix.append(tokenize(comment['body']))
                                reaction_matrix.append((1 if comment['reactions']['+1'] > 0 else 0,
                                                       1 if comment['reactions']['-1'] > 0 else 0,
                                                       1 if comment['reactions']['laugh'] > 0 else 0,
                                                       1 if comment['reactions']['hooray'] > 0 else 0,
                                                       1 if comment['reactions']['confused'] > 0 else 0,
                                                       1 if comment['reactions']['heart'] > 0 else 0))
                        for comment in user_data['review_comments']:
                            if comment['reactions']['total_count'] > 0:
                                reaction_count += 1
                                token_matrix.append(tokenize(comment['body']))
                                reaction_matrix.append((1 if comment['reactions']['+1'] > 0 else 0,
                                                        1 if comment['reactions']['-1'] > 0 else 0,
                                                        1 if comment['reactions']['laugh'] > 0 else 0,
                                                        1 if comment['reactions']['hooray'] > 0 else 0,
                                                        1 if comment['reactions']['confused'] > 0 else 0,
                                                        1 if comment['reactions']['heart'] > 0 else 0))
                        for comment in user_data['commit_comments']:
                            if comment['reactions']['total_count'] > 0:
                                reaction_count += 1
                                token_matrix.append(tokenize(comment['body']))
                                reaction_matrix.append((1 if comment['reactions']['+1'] > 0 else 0,
                                                        1 if comment['reactions']['-1'] > 0 else 0,
                                                        1 if comment['reactions']['laugh'] > 0 else 0,
                                                        1 if comment['reactions']['hooray'] > 0 else 0,
                                                        1 if comment['reactions']['confused'] > 0 else 0,
                                                        1 if comment['reactions']['heart'] > 0 else 0))

    print("Number of comments with reactions:", reaction_count, "Total number of comments:", total_count, "Fraction:", reaction_count/total_count)
    return token_matrix, np.array(reaction_matrix)
