from gensim.models import Phrases, phrases
from lib.util import mediawiki
from lib.data import fetch
import numpy as np


def train(train_x, min_count=30, threshold=150, max_vocab_size=50000000, scoring='default'):
    bigram_model = Phrases(train_x, min_count=min_count, threshold=threshold,
                           max_vocab_size=max_vocab_size, scoring=scoring)
    bigram_phraser = phrases.Phraser(bigram_model)
    trigram_model = Phrases(bigram_phraser[train_x])
    bigram_model.save('data/ngram/bigram_th%d_min%d' % (threshold, min_count))
    trigram_model.save('data/ngram/trigram_th%d_min%d' % (threshold, min_count))
    return bigram_model, trigram_model


def transform(bigram_model, trigram_model, transform_x):
    bigram_phraser = phrases.Phraser(bigram_model)
    trigram_phraser = phrases.Phraser(trigram_model)
    return trigram_phraser[bigram_phraser[transform_x]]


def load(bigram_model_path='data/ngram/bigram_th150_min20', trigram_model_path='data/ngram/trigram_th150_min20'):
    bigram_model = Phrases.load(bigram_model_path)
    trigram_model = Phrases.load(trigram_model_path)
    return bigram_model, trigram_model


if __name__ == '__main__':
    wiki_x = mediawiki.fetch()
    github_x = fetch.complete_text()
    train_x = np.concatenate([wiki_x, github_x])
    print(train_x.shape)
    bigram_model, trigram_model = train(train_x)
    # bigram_model, trigram_model = load()
    bigram_phraser = phrases.Phraser(bigram_model)
    trigram_phraser = phrases.Phraser(trigram_model)
    # for x in trigram_phraser[bigram_phraser[train_x]]:
    #     print(x)

    bigram_phrases = set()
    trigram_phrases = set()

    for phrase in bigram_model.export_phrases(train_x):
        if phrase[1] > 200:
            bigram_phrases.add(phrase)

    for phrase in trigram_model.export_phrases(train_x):
        if phrase[1] > 200:
            trigram_phrases.add(phrase)

    for phrase in bigram_phrases | trigram_phrases:
        print(phrase)
