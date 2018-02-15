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


def load():
    pass


def profile_devs(user_data):
    pass

if __name__ == '__main__':
    preprocess()