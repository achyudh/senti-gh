import wikipedia, datetime
from tinydb import TinyDB
from nltk.tokenize import word_tokenize, sent_tokenize
from lib.util import db
from wikipedia.exceptions import PageError

wikipedia.set_rate_limiting(200, min_wait=datetime.timedelta(0, 0, 500000))


def fetch(dataset_path, tokenize_words=True, tokenize_sentences=True):
    db = TinyDB(dataset_path)
    return db.all()


def download(page_ids, db_path):
    for page_id in page_ids:
        if len(db.query_wiki(page_id, db_path)) == 0:
            content = dict()
            try:
                page = wikipedia.page(pageid=page_id)
                content["id"] = page.pageid
                content["title"] = page.title
                content["content"] = page.content
                db.insert_repo(content, db_path=db_path)
            except AttributeError:
                pass
            except PageError:
                pass


if __name__ == '__main__':
    # with open("./data/wikipedia/titles/computer_programming.csv", 'r', encoding="utf8") as dataset_csv:
    #     _first_line = dataset_csv.readline()
    #     page_ids = list()
    #     for line in dataset_csv:
    #         page_ids.append(line.split(',')[1])
    #     download(page_ids, db_path="./data/wikipedia/content/computer_programming.json")
    #
    # print(len(fetch("./data/wikipedia/content/computer_programming.json")))

    # with open("./data/wikipedia/titles/software.csv", 'r', encoding="utf8") as dataset_csv:
    #     _first_line = dataset_csv.readline()
    #     page_ids = list()
    #     for line in dataset_csv:
    #         page_ids.append(line.split(',')[1])
    #     download(page_ids, db_path="./data/wikipedia/content/software.json")
    # print(len(fetch("./data/wikipedia/content/software.json")))

    with open("./data/wikipedia/titles/software_engineering.csv", 'r', encoding="utf8") as dataset_csv:
        _first_line = dataset_csv.readline()
        page_ids = list()
        for line in dataset_csv:
            page_ids.append(line.split(',')[1])
        download(page_ids, db_path="./data/wikipedia/content/software_engineering.json")
    print(len(fetch("./data/wikipedia/content/software_engineering.json")))