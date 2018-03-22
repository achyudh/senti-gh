from nltk.tokenize import word_tokenize
from lib.util import preprocessing, ngram
from gensim.models import phrases
from tinydb import TinyDB
import matplotlib.pyplot as plt
import json


def preprocess(raw_dataset="data/epa/word_dictionary_raw.csv", processed_dataset="data/epa/word_dictionary.json"):
    preprocessed_data = dict()
    with open(raw_dataset, 'r') as dataset_csv:
        _first_line = dataset_csv.readline()
        for line in dataset_csv:
            line = [x.strip() for x in line.split(',')]
            line[3] = line[3].lower().replace(' ', '_')
            if '/' in line[3]:
                for phrase in [x.strip() for x in line[3].split('/')]:
                    if preprocessed_data.get(phrase, None) is None:
                        preprocessed_data[phrase] = dict()
                        preprocessed_data[phrase]['Evaluation'] = list()
                        preprocessed_data[phrase]['Potency'] = list()
                        preprocessed_data[phrase]['Activity'] = list()
                        preprocessed_data[phrase]['Type'] = line[2]
                    if line[4] != '':
                        preprocessed_data[phrase]['Evaluation'].append(float(line[4]))
                    if line[5] != '':
                        preprocessed_data[phrase]['Potency'].append(float(line[5]))
                    if line[6] != '':
                        preprocessed_data[phrase]['Activity'].append(float(line[6]))
            else:
                if preprocessed_data.get(line[3], None) is None:
                    preprocessed_data[line[3]] = dict()
                    preprocessed_data[line[3]]['Evaluation'] = list()
                    preprocessed_data[line[3]]['Potency'] = list()
                    preprocessed_data[line[3]]['Activity'] = list()
                    preprocessed_data[line[3]]['Type'] = line[2]
                if line[4] != '':
                    preprocessed_data[line[3]]['Evaluation'].append(float(line[4]))
                if line[5] != '':
                    preprocessed_data[line[3]]['Potency'].append(float(line[5]))
                if line[6] != '':
                    preprocessed_data[line[3]]['Activity'].append(float(line[6]))

    for key, value in preprocessed_data.items():
        value['Avg_Evaluation'] = sum(value['Evaluation'])/len(value['Evaluation'])
        value['Avg_Potency'] = sum(value['Potency'])/len(value['Potency'])
        value['Avg_Activity'] = sum(value['Activity'])/len(value['Activity'])

    with open(processed_dataset, 'w') as dataset_json:
        json.dump(preprocessed_data, dataset_json)


def load(dataset="data/epa/word_dictionary.json"):
    with open(dataset, 'r') as dataset_json:
        return json.load(dataset_json)


def profile_devs(dataset, user_ipa):
    dev_profiles = dict()
    word_dict = load()
    db = TinyDB(dataset)
    bigram_model, trigram_model = ngram.load()
    bigram_phraser = phrases.Phraser(bigram_model)
    with open('data/epa/dev_profiles/values.csv', 'w') as csv_file:
        csv_file.write("user_id,a_prop,b_prop,c_prop,d_prop,epa_values" + "\n")
        for entry in db:
            for user_id, user_data in entry.items():
                if user_id in user_ipa:
                    if dev_profiles.get(user_id, None) is None:
                        dev_profiles[user_id] = dict()
                        dev_profiles[user_id]['Evaluation'] = list()
                        dev_profiles[user_id]['Potency'] = list()
                        dev_profiles[user_id]['Activity'] = list()
                    for value in user_data['issues']:
                        text = ''
                        if value['title'] is not None and value['body'] is not None:
                            text = word_tokenize((value['title'] + ' ' + value['body']).lower())
                        elif value['body'] is not None:
                            text = word_tokenize(value['body'].lower())
                        elif value['title'] is not None:
                            text = word_tokenize(value['title'].lower())
                        text = bigram_phraser[text]
                        for word in text:
                            if word in word_dict:
                                dev_profiles[user_id]['Evaluation'].append(word_dict[word]['Avg_Evaluation'])
                                dev_profiles[user_id]['Potency'].append(word_dict[word]['Avg_Potency'])
                                dev_profiles[user_id]['Activity'].append(word_dict[word]['Avg_Activity'])

                    for value in user_data['issue_comments']:
                        if value['body'] is not None:
                            text = word_tokenize(value['body'].lower())
                            text = bigram_phraser[text]
                            for word in text:
                                if word in word_dict:
                                    dev_profiles[user_id]['Evaluation'].append(word_dict[word]['Avg_Evaluation'])
                                    dev_profiles[user_id]['Potency'].append(word_dict[word]['Avg_Potency'])
                                    dev_profiles[user_id]['Activity'].append(word_dict[word]['Avg_Activity'])

                    for value in user_data['pull_requests']:
                        text = ''
                        if value['title'] is not None and value['body'] is not None:
                            text = word_tokenize((value['title'] + ' ' + value['body']).lower())
                        elif value['body'] is not None:
                            text = word_tokenize(value['body'].lower())
                        elif value['title'] is not None:
                            text = word_tokenize(value['title'].lower())
                        text = bigram_phraser[text]
                        for word in text:
                            if word in word_dict:
                                dev_profiles[user_id]['Evaluation'].append(word_dict[word]['Avg_Evaluation'])
                                dev_profiles[user_id]['Potency'].append(word_dict[word]['Avg_Potency'])
                                dev_profiles[user_id]['Activity'].append(word_dict[word]['Avg_Activity'])

                    for value in user_data['review_comments']:
                        if value['body'] is not None:
                            text = word_tokenize(value['body'].lower())
                            text = bigram_phraser[text]
                            for word in text:
                                if word in word_dict:
                                    dev_profiles[user_id]['Evaluation'].append(word_dict[word]['Avg_Evaluation'])
                                    dev_profiles[user_id]['Potency'].append(word_dict[word]['Avg_Potency'])
                                    dev_profiles[user_id]['Activity'].append(word_dict[word]['Avg_Activity'])

                    for value in user_data['commits']:
                        if value['message'] is not None:
                            text = word_tokenize(value['message'].lower())
                            text = bigram_phraser[text]
                            for word in text:
                                if word in word_dict:
                                    dev_profiles[user_id]['Evaluation'].append(word_dict[word]['Avg_Evaluation'])
                                    dev_profiles[user_id]['Potency'].append(word_dict[word]['Avg_Potency'])
                                    dev_profiles[user_id]['Activity'].append(word_dict[word]['Avg_Activity'])

                    for value in user_data['commit_comments']:
                        if value['body'] is not None:
                            text = word_tokenize(value['body'].lower())
                            text = bigram_phraser[text]
                            for word in text:
                                if word in word_dict:
                                    dev_profiles[user_id]['Evaluation'].append(word_dict[word]['Avg_Evaluation'])
                                    dev_profiles[user_id]['Potency'].append(word_dict[word]['Avg_Potency'])
                                    dev_profiles[user_id]['Activity'].append(word_dict[word]['Avg_Activity'])

                    if len(dev_profiles[user_id]['Evaluation']) > 80:
                        # dev_profiles[user_id]['Evaluation'] = [x/sum(dev_profiles[user_id]['Evaluation']) for x in dev_profiles[user_id]['Evaluation']]
                        # dev_profiles[user_id]['Potency'] = [x/sum(dev_profiles[user_id]['Potency']) for x in dev_profiles[user_id]['Potency']]
                        # dev_profiles[user_id]['Activity'] = [x/sum(dev_profiles[user_id]['Activity']) for x in dev_profiles[user_id]['Activity']]

                        fig, ax = plt.subplots(nrows=1, ncols=3)
                        plt.figure(figsize=(30, 10))
                        plt.subplot(1, 3, 1)
                        e_counts, e_bins, e_bars = plt.hist(dev_profiles[user_id]['Evaluation'], bins=20, normed=True, range=[-1, 3])
                        plt.ylabel('Evaluation for ' + user_id)

                        plt.subplot(1, 3, 2)
                        p_counts, p_bins, p_bars = plt.hist(dev_profiles[user_id]['Potency'], bins=20, normed=True, range=[-1, 3])
                        plt.ylabel('Potency for ' + user_id)

                        plt.subplot(1, 3, 3)
                        a_counts, a_bins, a_bars = plt.hist(dev_profiles[user_id]['Activity'], bins=20, normed=True, range=[-1, 3])
                        plt.ylabel('Activity for ' + user_id)

                        plt.savefig('data/epa/dev_profiles/%s' % user_id)
                        csv_file.write(user_id + "," + str(user_ipa[user_id]['a']) + ',' + str(user_ipa[user_id]['b'])
                                       + ',' + str(user_ipa[user_id]['c']) + ',' + str(user_ipa[user_id]['d'])
                                       + ',' + ",".join(str(x) for x in e_counts) + ',' + ",".join(str(x) for x in p_counts)
                                       + ',' + ",".join(str(x) for x in a_counts) + "\n")
                        plt.close()


if __name__ == '__main__':
    preprocess()
    user_ipa = preprocessing.user_ipa_count("data/labelled/pull_requests/oni.json")
    profile_devs("data/user/onivim/oni.json", user_ipa)