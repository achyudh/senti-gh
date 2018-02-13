from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from lib.embedding.vectorizer import HybridEmbeddingVectorizer
from lib.embedding.word2vec import train_gensim, load_gensim
from lib.data import fetch
import numpy as np


def train(train_x, train_y, w2v1, w2v2, dim1, dim2):
    rfc_pipeline = Pipeline([
        ("TfidfEmbeddingVectorizer", HybridEmbeddingVectorizer(w2v1, w2v2, dim1, dim2)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=500, n_jobs=8))])
    rfc_pipeline.fit(train_x, train_y)
    return rfc_pipeline


def predict(classifier, predict_x):
    return classifier.predict(predict_x)


def evaluate(classifier, evaluate_x, evaluate_y):
    predict_y = classifier.predict(evaluate_x)
    return {"individual":precision_recall_fscore_support(evaluate_y, predict_y),
            "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}


def cross_val(data_x, data_y, w2v1, w2v2, dim1, dim2, n_splits=5):
    skf = StratifiedKFold(n_splits)
    print("Performing cross validation (%d-fold)..." % n_splits)
    print("Precision, Recall, F_Score, Support")
    mean_precision = 0
    mean_recall = 0
    mean_accuracy = 0
    for train_index, test_index in skf.split(data_x, data_y):
        rfc_pipeline = train(data_x[train_index], data_y[train_index], w2v1, w2v2, dim1, dim2)
        metrics = evaluate(rfc_pipeline, data_x[test_index], data_y[test_index])
        mean_precision += metrics['individual'][0][1]
        mean_recall += metrics['individual'][1][1]
        mean_accuracy += metrics['micro-average'][0]
        print(metrics)
    print("Mean precision: %s, Mean recall: %s Mean accuracy: %s" % (mean_precision/n_splits, mean_recall/n_splits, mean_accuracy/n_splits))


if __name__ == '__main__':
    data_x, reaction_matrix = fetch.text_with_reactions("data/user")
    data_x = np.array(data_x)
    data_y = reaction_matrix[:, 0]
    print("Reaction skew for +1", sum(reaction_matrix[:, 0]) / len(reaction_matrix[:, 0]))
    w2v1 = load_gensim('data/embedding/word2vec/gensim_size300_min5')
    w2v2 = load_gensim('data/embedding/word2vec/googlenews_size300.bin', binary=True)
    cross_val(data_x, data_y, w2v1, w2v2, 300, 300, n_splits=5)