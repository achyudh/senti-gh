from tinydb import TinyDB, Query


def insert(payload, full_repo_name):
    db = TinyDB('data/%s.json' % full_repo_name)
    return db.insert(payload)


def query(user_login, full_repo_name):
    query_obj = Query()
    db = TinyDB('data/%s.json' % full_repo_name)
    return db.search(query_obj.login == user_login)