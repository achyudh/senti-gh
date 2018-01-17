from tinydb import TinyDB, Query


def insert(payload, db_path):
    db = TinyDB(db_path)
    return db.insert(payload)


def query_user(user_login, full_repo_name):
    query_obj = Query()
    db = TinyDB('data/user/%s.json' % full_repo_name)
    return db.search(query_obj.login == user_login)