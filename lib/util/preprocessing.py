from tinydb import TinyDB


def user_ipa_count(dataset):
    result = dict()
    db = TinyDB(dataset)
    for entry in db:
        for thread in entry.values():
            for comment in thread['comments']:
                if 'ipa' in comment:
                    user_login = comment['user']['login']
                    category = comment['ipa']
                    if user_login not in result:
                        result[user_login] = dict()
                        result[user_login]['a'] = 0
                        result[user_login]['b'] = 0
                        result[user_login]['c'] = 0
                        result[user_login]['d'] = 0
                    if category != '-':
                        result[user_login][category.lower()] += 1
    for v in result.values():
        sum = v['a'] + v['b'] + v['c'] + v['d']
        if sum != 0:
            v['a'] = v['a']/sum
            v['b'] = v['b']/sum
            v['c'] = v['c']/sum
            v['d'] = v['d']/sum
    return result

if __name__ == '__main__':
    user_ipa = user_ipa_count("data/labelled/pull_requests/tensorflow.json")
