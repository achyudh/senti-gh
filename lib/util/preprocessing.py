

def user_ipa_count(dataset):
    result = dict()
    with open(dataset) as csv_file:
        _first_line = csv_file.readline()
        for line in csv_file:
            split_line = line.split(',')
            if split_line[4] not in result:
                result[split_line[4]] = dict()
                result[split_line[4]]['a'] = 0
                result[split_line[4]]['b'] = 0
                result[split_line[4]]['c'] = 0
                result[split_line[4]]['d'] = 0
            for category in split_line[1]:
                if category != '-':
                    result[split_line[4]][category.lower()] += 1
    for v in result.values():
        sum = v['a'] + v['b'] + v['c'] + v['d']
        if sum != 0:
            v['a'] = v['a']/sum
            v['b'] = v['b']/sum
            v['c'] = v['c']/sum
            v['d'] = v['d']/sum
    return result

