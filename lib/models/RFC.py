from lib.data import fetch
from lib.embeddings.glove import MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer, glove6B
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def train(train_x, train_y):
    w2v = glove6B()
    rfc_pipeline = Pipeline([
        ("MeanEmbeddingVectorizer", MeanEmbeddingVectorizer(w2v)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=200, n_jobs=7))])
    rfc_pipeline.fit(train_x, train_y)
    return rfc_pipeline


def predict(classifier, predict_x):
    return classifier.predict(predict_x)


def evaluate(classifier, evaluate_x, evaluate_y):
    predict_y = classifier.predict(evaluate_x)
    return precision_recall_fscore_support(evaluate_y, predict_y)


def cross_val(data_x, data_y, n_splits=5):
    skf = StratifiedKFold(n_splits)
    print("Performing cross validation (%d fold)..." % n_splits)
    for train_index, test_index in skf.split(data_x, data_y):
        rfc_pipeline = train(data_x[train_index], data_y[train_index])
        print(evaluate(rfc_pipeline, data_x[test_index], data_y[test_index]))
    print("Precision, Recall, F_Score, Support")


if __name__ == '__main__':
    data_x, data_y = fetch.labelled_comments("./data/labelled/pull_requests/grouped_emotions.csv")
    cross_val(data_x, data_y)