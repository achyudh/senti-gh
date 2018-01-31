from lib.embedding.glove import TfidfEmbeddingVectorizer, load_glove6B
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from lib.embedding.word2vec import train_gensim, load_gensim
from sklearn.pipeline import Pipeline
from lib.data import fetch
import numpy as np

def train(train_x, train_y, w2v):
    rfc_pipeline = Pipeline([
        ("TfidfEmbeddingVectorizer", TfidfEmbeddingVectorizer(w2v)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=300, n_jobs=8))])
    rfc_pipeline.fit(train_x, train_y)
    return rfc_pipeline


def predict(classifier, predict_x):
    return classifier.predict(predict_x)


def evaluate(classifier, evaluate_x, evaluate_y):
    predict_y = classifier.predict(evaluate_x)
    return {"individual":precision_recall_fscore_support(evaluate_y, predict_y),
            "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}


def cross_val(data_x, data_y, w2v, n_splits=5):
    skf = StratifiedKFold(n_splits)
    print("Performing cross validation (%d-fold)..." % n_splits)
    print("Precision, Recall, F_Score, Support")
    for train_index, test_index in skf.split(data_x, data_y):
        rfc_pipeline = train(data_x[train_index], data_y[train_index], w2v)
        print(evaluate(rfc_pipeline, data_x[test_index], data_y[test_index]))


if __name__ == '__main__':
    data_x, data_y = fetch.labelled_comments("./data/labelled/pull_requests/agrees.csv")
    # w2v = train_gensim(np.concatenate([data_x, fetch.complete_text()]), size=300)
    w2v = load_gensim()
    print("Agrees:")
    cross_val(data_x, data_y, w2v, n_splits=5)
    # data_x, data_y = fetch.labelled_comments("./data/labelled/pull_requests/agrees_further.csv")
    # print("Agrees further:")
    # cross_val(data_x, data_y, w2v, n_splits=5)
    data_x, data_y = fetch.labelled_comments("./data/labelled/pull_requests/gives_opinion.csv")
    print("Gives opinion:")
    cross_val(data_x, data_y, w2v, n_splits=5)
    data_x, data_y = fetch.labelled_comments("./data/labelled/pull_requests/grouped_emotions.csv")
    print("Grouped emotions:")
    cross_val(data_x, data_y, w2v, n_splits=5)
