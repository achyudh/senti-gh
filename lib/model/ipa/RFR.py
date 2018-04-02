from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def train(train_x, train_y):
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(train_x, train_y)
    return regressor



def predict(regressor, predict_x):
    return regressor.predict(predict_x)


def evaluate(regressor, evaluate_x, evaluate_y):
    return regressor.score(evaluate_x, evaluate_y)


def cross_val(data_x, data_y, n_splits=5):
    skf = KFold(n_splits)
    print("Performing cross validation (%d-fold)..." % n_splits)
    for train_index, test_index in skf.split(data_x, data_y):
        regressor = train(data_x[train_index], data_y[train_index])
        metrics = evaluate(regressor, data_x[test_index], data_y[test_index])
        print(metrics)

if __name__ == '__main__':
    data_x = list()
    data_ya = list()
    data_yb = list()
    data_yc = list()
    data_yd = list()
    with open("data/epa/dev_profiles/values_all.csv") as csv_file:
        _first_line = csv_file.readline()
        for line in csv_file:
            split_line = line.split(',')
            data_x.append(split_line[5:])
            data_ya.append(split_line[1])
            data_yb.append(split_line[2])
            data_yc.append(split_line[3])
            data_yd.append(split_line[4])
    data_x = np.array(data_x).astype(np.float)
    data_ya = np.array(data_ya).astype(np.float).reshape(-1, 1)
    data_yb = np.array(data_yb).astype(np.float).reshape(-1, 1)
    data_yc = np.array(data_yc).astype(np.float).reshape(-1, 1)
    data_yd = np.array(data_yd).astype(np.float).reshape(-1, 1)
    print("A:")
    cross_val(data_x, data_ya, n_splits=5)
    print("B:")
    cross_val(data_x, data_yb, n_splits=5)
    print("C:")
    cross_val(data_x, data_yc, n_splits=5)
    print("D:")
    cross_val(data_x, data_yd, n_splits=5)