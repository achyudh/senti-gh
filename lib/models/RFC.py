from lib.data import fetch
from lib.embeddings.pre_trained import MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer, glove6B
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def train(train_x, train_y):
    w2v = glove6B()
    rfc_pipeline = Pipeline([
        ("MeanEmbeddingVectorizer", MeanEmbeddingVectorizer(w2v)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=200, n_jobs=7))])
    rfc_pipeline.fit(train_x, train_y)
    return rfc_pipeline


def predict(classifier, predict_x):
    pass


def evaluate(classifier, evaluate_x, evaluate_y):
    pass


def cross_val(x, y):
    rfc_pipeline = train()


data_x, data_y = fetch.labelled_comments("./data/labelled/pull_requests/grouped_emotions.csv")
cross_val()