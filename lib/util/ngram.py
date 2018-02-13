from gensim.models import Phrases
from lib.util import mediawiki
from lib.data.fetch import complete_text
import numpy as np

def train(train_x, min_count=10, threshold=200, max_vocab_size=40000000, scoring='default'):
    bigram_model = Phrases(train_x, min_count=min_count, threshold=threshold,
                           max_vocab_size=max_vocab_size, scoring=scoring)
    trigram_model = Phrases(bigram_model[train_x])
    bigram_model.save('data/ngram/bigram_th%d_min%d' % (threshold, min_count))
    trigram_model.save('data/ngram/trigram_th%d_min%d' % (threshold, min_count))
    return bigram_model, trigram_model


def transform(bigram_model, trigram_model, transform_x):
    return trigram_model[bigram_model[transform_x]]


def load(bigram_model_path='data/ngram/bigram', trigram_model_path='data/ngram/trigram'):
    bigram_model = Phrases.load(bigram_model_path)
    trigram_model = Phrases.load(trigram_model_path)
    return bigram_model, trigram_model


if __name__ == '__main__':
    train_x = mediawiki.fetch()
    print(train_x.shape)
    bigram_model, trigram_model = train(train_x)
    # for line in transform(bigram_model, trigram_model, train_x):
    #     print(' '.join(line))

    bigram_phrases = set()
    trigram_phrases = set()

    for phrase in bigram_model.export_phrases(train_x):
        bigram_phrases.add(phrase)

    for phrase in trigram_model.export_phrases(train_x):
        trigram_phrases.add(phrase)

    for phrase in bigram_phrases | trigram_phrases:
        print(phrase)
