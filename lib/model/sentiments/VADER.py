from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import pandas as pd
import numpy as np


def predict(predict_x, has_neutral):
    sid = SentimentIntensityAnalyzer()
    predict_y = list()
    for paragraph in predict_x:
        polarity = 0
        for sentence in tokenize.sent_tokenize(paragraph):
            polarity += sid.polarity_scores(sentence)['compound']
        if has_neutral:
            if polarity > 0:
                predict_y.append(1)
            elif polarity < 0:
                predict_y.append(0)
            else:
                predict_y.append(2)
        else:
            if polarity >= 0:
                predict_y.append(1)
            else:
                predict_y.append(0)
    return predict_y


def evaluate(evaluate_x, evaluate_y, has_neutral=False):
    predict_y = predict(evaluate_x, has_neutral)
    return {"individual": precision_recall_fscore_support(evaluate_y, predict_y),
            "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}


if __name__ == '__main__':
    data = pd.read_csv("data/labelled/Gerrit.csv").as_matrix()
    # data = pd.read_csv("data/labelled/StackOverflow.csv", encoding='latin1').as_matrix()
    data_x = np.array([x for x in data[:,0]])
    data_y = np.array([int(x) for x in data[:,1]])
    print("Dataset loaded to memory. Size:", len(data_y))
    print(evaluate(data_x, data_y, has_neutral=False))
