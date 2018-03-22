from tinydb import TinyDB, Query
import pathlib


def insert_generic(payload, db_path):
    db = TinyDB(db_path)
    return db.insert(payload)


def insert_repo(payload, db_path):
    repo_name = db_path.split('/')[2]
    pathlib.Path('data/repo/' + repo_name).mkdir(parents=True, exist_ok=True)
    pathlib.Path('data/user/' + repo_name).mkdir(parents=True, exist_ok=True)
    db = TinyDB(db_path)
    return db.insert(payload)


def query_repo(full_repo_name):
    db = TinyDB('data/repo/%s.json' % full_repo_name)
    return db.all()


def query_user(user_login, full_repo_name):
    query_obj = Query()
    db = TinyDB('data/user/%s.json' % full_repo_name)
    return db.search(query_obj.login == user_login)


def query_wiki(page_id, db_path):
    query_obj = Query()
    db = TinyDB(db_path)
    return db.search(query_obj.id == page_id)

