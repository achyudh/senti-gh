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


def evaluate_multiple(data_1, data_2, data_3, has_neutral=False):
    precision_list = [0 for i in range(3 if has_neutral else 2)]
    recall_list = [0 for i in range(3 if has_neutral else 2)]
    mean_accuracy = 0
    data_x1 = np.array([x for x in data_1[:,0]])
    data_y1 = np.array([int(x) for x in data_1[:,1]])
    data_x2 = np.array([x for x in data_2[:,0]])
    data_y2 = np.array([int(x) for x in data_2[:,1]])
    data_x3 = np.array([x for x in data_3[:,0]])
    data_y3 = np.array([int(x) for x in data_3[:,1]])

    metrics = evaluate(data_x1, data_y1, has_neutral=False)
    mean_accuracy += metrics['micro-average'][0]
    precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
    recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
    print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))

    metrics = evaluate(data_x2, data_y2, has_neutral=False)
    mean_accuracy += metrics['micro-average'][0]
    precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
    recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
    print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))

    metrics = evaluate(data_x3, data_y3, has_neutral=False)
    mean_accuracy += metrics['micro-average'][0]
    precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
    recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
    print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))

    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/3, [precision/3 for precision in precision_list], [recall/3 for recall in recall_list]))


if __name__ == '__main__':
    # data = pd.read_csv("data/labelled/Gerrit.csv").as_matrix()
    # data = pd.read_csv("data/labelled/StackOverflowcsv", encoding='latin1').as_matrix()
    data_1 = pd.read_csv("data/labelled/StackOverflow2.csv", encoding='latin1').as_matrix()
    data_2 = pd.read_csv("data/labelled/Gerrit.csv").as_matrix()
    data_3 = pd.read_csv("data/labelled/JIRA.csv").as_matrix()
    evaluate_multiple(data_1, data_2, data_3)
    # data = pd.concat([data_1, data_2, data_3]).as_matrix()
    # data_x = np.array([x for x in data[:,0]])
    # data_y = np.array([int(x) for x in data[:,1]])
    # print("Dataset loaded to memory. Size:", len(data_y))
    # metrics = evaluate(data_x, data_y, has_neutral=False)
    # print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))
