from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
import pandas as pd
import numpy as np
import subprocess


def train(train_x, train_y):
    with open("input.csv", 'w', encoding='utf-8') as text_file:
        for line, label in zip(train_x, train_y):
            text_file.write(' '.join(line.split()) + ',' + str(label) + '\n')

    subprocess.run(["java", "-jar", ".\\lib\\external\\Senti4SD\\NgramsExtraction.jar", "input.csv",
                    "true"]).check_returncode()
    subprocess.run(["java",  "-jar", "-Xmx5000m", ".\\lib\\external\\Senti4SD\\Senti4SD.jar", "-F", "A", "-i",
                    "input.csv", "-W", ".\\lib\\external\\Senti4SD\\dsm.bin", "-oc", "output.csv", "-vd", "600",
                    "-L", "-ul", "UnigramsList", "bl", "BigramsList"]).check_returncode()
    subprocess.run(["Rscript", ".\\lib\\external\\Senti4SD\\parameterTuning.R", "temp",
                    ".\\lib\\external\\Senti4SD\\modelsLiblinear", "output.csv"]).check_returncode()
    subprocess.run(["Rscript", ".\\lib\\external\\Senti4SD\\trainModel.R", "temp",
                    "temp\\L2-regularized_logistic_regression_(primal).txt", "output.csv"]).check_returncode()


def predict(predict_x):
    with open("test.txt", 'w', encoding='utf-8') as text_file:
        for line in predict_x:
            text_file.write(' '.join(line.split()) + '\n')

    subprocess.run(["java", "-jar", ".\\lib\\external\\Senti4SD\\NgramsExtraction.jar", "test.txt"]).check_returncode()
    subprocess.run(["java",  "-jar", "-Xmx5000m", ".\\lib\\external\\Senti4SD\\Senti4SD.jar", "-F", "A", "-i",
                    "test.txt", "-W", ".\\lib\\external\\Senti4SD\\dsm.bin", "-oc", "output.csv", "-vd", "600",
                    "-ul", "UnigramsList", "bl", "BigramsList"]).check_returncode()
    subprocess.run(["Rscript", ".\\lib\\external\\Senti4SD\\classification.R", "output.csv", "predictions.csv",
                    "temp\\modelLiblinear_L2-regularized_logistic_regression_(primal).Rda"]).check_returncode()

    predict_y = list()
    with open("predictions.csv", 'r') as text_file:
        _temp = text_file.readline()
        for line in text_file:
            predict_y.append(int(line.split(',')[1].strip()))
    return predict_y


def evaluate(evaluate_x, evaluate_y):
    predict_y = predict(evaluate_x)
    return {"individual": precision_recall_fscore_support(evaluate_y, predict_y),
            "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}


def cross_val(data_x, data_y, num_classes, n_splits=5):
    skf = StratifiedKFold(n_splits, random_state=157)
    print("Performing cross validation (%d-fold)..." % n_splits)
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    mean_accuracy = 0
    for train_index, test_index in skf.split(data_x, data_y):
        train(data_x[train_index], data_y[train_index])
        metrics = evaluate(data_x[test_index], data_y[test_index])
        precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
        recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
        mean_accuracy += metrics['micro-average'][0]
        print("Precision, Recall, F_Score, Support")
        print(metrics)
    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/n_splits, [precision/n_splits for precision in precision_list], [recall/n_splits for recall in recall_list]))


def bootstrap_trend(data_x, data_y):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=157)
    print("Metrics: Precision, Recall, F_Score, Support")
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    accuracy_list = list()
    for sample_rate in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        n_samples = int(sample_rate * len(train_y) + 1)
        train_xr, train_yr = resample(train_x, train_y, n_samples=n_samples, random_state=157)
        print("Training with %d samples" % len(train_yr))
        train(train_xr, train_yr)
        metrics = evaluate(test_x, test_y)
        precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
        recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
        accuracy_list.append(metrics['micro-average'][0])
        print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))
    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (sum(accuracy_list)/9, [precision/9 for precision in precision_list], [recall/9 for recall in recall_list]))
    print(accuracy_list)


def hard_cross_val(data_1, data_2, data_3, num_classes):
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    mean_accuracy = 0
    data_x1 = np.array([x.lower().split() for x in data_1.as_matrix()[:,0]])
    data_y1 = np.array([int(x) for x in data_1.as_matrix()[:,1]])
    data_x2 = np.array([x.lower().split() for x in data_2.as_matrix()[:,0]])
    data_y2 = np.array([int(x) for x in data_2.as_matrix()[:,1]])
    data_x3 = np.array([x.lower().split() for x in data_3.as_matrix()[:,0]])
    data_y3 = np.array([int(x) for x in data_3.as_matrix()[:,1]])

    data_12 = pd.concat([data_1, data_2]).as_matrix()
    data_x12 = np.array([x.lower().split() for x in data_12[:,0]])
    data_y12 = np.array([int(x) for x in data_12[:,1]])
    train(data_x12, data_y12)
    metrics = evaluate(data_x3, data_y3)
    mean_accuracy += metrics['micro-average'][0]
    precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
    recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
    print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))

    data_23 = pd.concat([data_2, data_3]).as_matrix()
    data_x23 = np.array([x.lower().split() for x in data_23[:,0]])
    data_y23 = np.array([int(x) for x in data_23[:,1]])
    train(data_x23, data_y23)
    metrics = evaluate(data_x1, data_y1)
    mean_accuracy += metrics['micro-average'][0]
    precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
    recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
    print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))

    data_13 = pd.concat([data_1, data_3]).as_matrix()
    data_x13 = np.array([x.lower().split() for x in data_13[:,0]])
    data_y13 = np.array([int(x) for x in data_13[:,1]])
    train(data_x13, data_y13)
    metrics = evaluate(data_x2, data_y2)
    mean_accuracy += metrics['micro-average'][0]
    precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
    recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
    print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))
    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/3, [precision/3 for precision in precision_list], [recall/3 for recall in recall_list]))


def evaluate_custom(data_1, data_2, data_3, num_classes):
    data_x1 = np.array([x.lower().split() for x in data_1.as_matrix()[:,0]])
    data_y1 = np.array([int(x) for x in data_1.as_matrix()[:,1]])
    data_x2 = np.array([x.lower().split() for x in data_2.as_matrix()[:,0]])
    data_y2 = np.array([int(x) for x in data_2.as_matrix()[:,1]])
    data_x3 = np.array([x.lower().split() for x in data_3.as_matrix()[:,0]])
    data_y3 = np.array([int(x) for x in data_3.as_matrix()[:,1]])
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    mean_accuracy = 0

    train(data_x1, data_y1)
    metrics = evaluate(data_x2, data_y2)
    mean_accuracy += metrics['micro-average'][0]
    precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
    recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
    print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))

    metrics = evaluate(data_x3, data_y3)
    mean_accuracy += metrics['micro-average'][0]
    precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
    recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
    print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))

    train(data_x2, data_y2)
    metrics = evaluate(data_x1, data_y1)
    mean_accuracy += metrics['micro-average'][0]
    precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
    recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
    print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))

    metrics = evaluate(data_x3, data_y3)
    mean_accuracy += metrics['micro-average'][0]
    precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
    recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
    print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))

    train(data_x3, data_y3)
    metrics = evaluate(data_x1, data_y1)
    mean_accuracy += metrics['micro-average'][0]
    precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
    recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
    print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))

    metrics = evaluate(data_x2, data_y2)
    mean_accuracy += metrics['micro-average'][0]
    precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
    recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
    print("Accuracy: %s Precision: %s, Recall: %s" % (metrics['micro-average'][0], metrics['individual'][0], metrics['individual'][1]))

    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/6, [precision/6 for precision in precision_list], [recall/6 for recall in recall_list]))


if __name__ == '__main__':
    num_classes = 2
    data = pd.read_csv("data/labelled/JIRA.csv").as_matrix()
    # data = pd.read_csv("data/labelled/StackOverflow.csv", encoding='latin1').as_matrix()
    # data_1 = pd.read_csv("data/labelled/Gerrit.csv")
    # data_2 = pd.read_csv("data/labelled/JIRA.csv")
    # data_3 = pd.read_csv("data/labelled/StackOverflow2.csv", encoding='latin1')
    # hard_cross_val(data_1, data_2, data_3, num_classes)
    # evaluate_custom(data_1, data_2, data_3, num_classes)
    # data = pd.concat([data_1, data_2, data_3]).as_matrix()
    data_x = np.array([x.lower() for x in data[:,0]])
    data_y = np.array([int(x) for x in data[:,1]])
    # print("Dataset loaded to memory. Size:", len(data_y))
    cross_val(data_x, data_y, num_classes, n_splits=5)
    # bootstrap_trend(data_x, data_y)