from nltk.tokenize import word_tokenize, sent_tokenize
from wikipedia.exceptions import PageError
from gensim.models import phrases
from lib.util import db, ngram
from tinydb import TinyDB
import wikipedia, datetime, os
import numpy as np

wikipedia.set_rate_limiting(200, min_wait=datetime.timedelta(0, 0, 500000))


def fetch(dataset_path="data/wikipedia/content", tokenize_words=True, tokenize_sentences=True, detect_ngrams=False):
    token_matrix = list()
    for filename in os.listdir(dataset_path):
        if filename.endswith(".json"):
            db = TinyDB(os.path.join(dataset_path, filename))
            for entry in db:
                if tokenize_sentences and tokenize_words:
                    token_matrix.extend([word_tokenize(x.lower()) for x in sent_tokenize(entry['content'])])
                elif tokenize_sentences and not tokenize_words:
                    token_matrix.extend([x.lower() for x in sent_tokenize(entry['content'])])
                elif not tokenize_sentences and tokenize_words:
                    token_matrix.extend(word_tokenize(entry['content'].lower()))
                else:
                    token_matrix.append(entry['content'].lower())
    if detect_ngrams:
        bigram_model, trigram_model = ngram.load()
        bigram_phraser = phrases.Phraser(bigram_model)
        trigram_phraser = phrases.Phraser(trigram_model)
        return trigram_phraser[bigram_phraser[token_matrix]]
    else:
        return np.array(token_matrix)


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

    # with open("./data/wikipedia/titles/software.csv", 'r', encoding="utf8") as dataset_csv:
    #     _first_line = dataset_csv.readline()
    #     page_ids = list()
    #     for line in dataset_csv:
    #         page_ids.append(line.split(',')[1])
    #     download(page_ids, db_path="./data/wikipedia/content/software.json")

    # with open("./data/wikipedia/titles/software_engineering.csv", 'r', encoding="utf8") as dataset_csv:
    #     _first_line = dataset_csv.readline()
    #     page_ids = list()
    #     for line in dataset_csv:
    #         page_ids.append(line.split(',')[1])
    #     download(page_ids, db_path="./data/wikipedia/content/software_engineering.json")

    print(fetch().shape)