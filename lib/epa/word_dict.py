from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from tinydb import TinyDB
import json


def preprocess(raw_dataset="data/epa/word_dictionary_raw.csv", processed_dataset="data/epa/word_dictionary.json"):
    preprocessed_data = dict()
    with open(raw_dataset, 'r') as dataset_csv:
        _first_line = dataset_csv.readline()
        for line in dataset_csv:
            line = [x.strip() for x in line.split(',')]
            line[3] = line[3].lower()
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


def profile_devs(dataset):
    dev_profiles = dict()
    word_dict = load()
    db = TinyDB(dataset)
    for entry in db:
        for user_id, user_data in entry.items():
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
                for word in text:
                    if word in word_dict:
                        dev_profiles[user_id]['Evaluation'].append(word_dict[word]['Avg_Evaluation'])
                        dev_profiles[user_id]['Potency'].append(word_dict[word]['Avg_Potency'])
                        dev_profiles[user_id]['Activity'].append(word_dict[word]['Avg_Activity'])

            for value in user_data['issue_comments']:
                if value['body'] is not None:
                    text = word_tokenize(value['body'].lower())
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
                for word in text:
                    if word in word_dict:
                        dev_profiles[user_id]['Evaluation'].append(word_dict[word]['Avg_Evaluation'])
                        dev_profiles[user_id]['Potency'].append(word_dict[word]['Avg_Potency'])
                        dev_profiles[user_id]['Activity'].append(word_dict[word]['Avg_Activity'])

            for value in user_data['review_comments']:
                if value['body'] is not None:
                    text = word_tokenize(value['body'].lower())
                    for word in text:
                        if word in word_dict:
                            dev_profiles[user_id]['Evaluation'].append(word_dict[word]['Avg_Evaluation'])
                            dev_profiles[user_id]['Potency'].append(word_dict[word]['Avg_Potency'])
                            dev_profiles[user_id]['Activity'].append(word_dict[word]['Avg_Activity'])

            for value in user_data['commits']:
                if value['message'] is not None:
                    text = word_tokenize(value['message'].lower())
                    for word in text:
                        if word in word_dict:
                            dev_profiles[user_id]['Evaluation'].append(word_dict[word]['Avg_Evaluation'])
                            dev_profiles[user_id]['Potency'].append(word_dict[word]['Avg_Potency'])
                            dev_profiles[user_id]['Activity'].append(word_dict[word]['Avg_Activity'])

            for value in user_data['commit_comments']:
                if value['body'] is not None:
                    text = word_tokenize(value['body'].lower())
                    for word in text:
                        if word in word_dict:
                            dev_profiles[user_id]['Evaluation'].append(word_dict[word]['Avg_Evaluation'])
                            dev_profiles[user_id]['Potency'].append(word_dict[word]['Avg_Potency'])
                            dev_profiles[user_id]['Activity'].append(word_dict[word]['Avg_Activity'])

            if len(dev_profiles[user_id]['Evaluation']) > 100:
                dev_profiles[user_id]['Evaluation'] = [x/sum(dev_profiles[user_id]['Evaluation']) for x in dev_profiles[user_id]['Evaluation']]
                dev_profiles[user_id]['Potency'] = [x/sum(dev_profiles[user_id]['Potency']) for x in dev_profiles[user_id]['Potency']]
                dev_profiles[user_id]['Activity'] = [x/sum(dev_profiles[user_id]['Activity']) for x in dev_profiles[user_id]['Activity']]

                fig, ax = plt.subplots(nrows=1, ncols=3)
                plt.figure(figsize=(30, 10))
                plt.subplot(1, 3, 1)
                plt.hist(dev_profiles[user_id]['Evaluation'], bins=20)
                plt.ylabel('Evaluation for ' + user_id)

                plt.subplot(1, 3, 2)
                plt.hist(dev_profiles[user_id]['Potency'], bins=20)
                plt.ylabel('Potency for ' + user_id)

                plt.subplot(1, 3, 3)
                plt.hist(dev_profiles[user_id]['Activity'], bins=20)
                plt.ylabel('Activity for ' + user_id)

                plt.savefig('data/epa/dev_profiles/%s' % user_id)
                plt.close()


if __name__ == '__main__':
    profile_devs("data/user/facebook/hhvm.json")