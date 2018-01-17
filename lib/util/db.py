from tinydb import TinyDB, Query
import pathlib

def insert(payload, db_path):
    repo_name = db_path.split('/')[2]
    pathlib.Path('data/repo/' + repo_name).mkdir(parents=True, exist_ok=True)
    pathlib.Path('data/user/' + repo_name).mkdir(parents=True, exist_ok=True)
    db = TinyDB(db_path)
    return db.insert(payload)


def query_user(user_login, full_repo_name):
    query_obj = Query()
    db = TinyDB('data/user/%s.json' % full_repo_name)
    return db.search(query_obj.login == user_login)