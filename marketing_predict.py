#
# credit to Li Xinyu
#
import csv

import numpy
import pandas as pd
from random import uniform
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn import tree, svm
from sklearn.metrics import make_scorer, matthews_corrcoef, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier


from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids, NearMiss
from imblearn.combine import SMOTEENN
import category_encoders as ce
from imba_algo.smote_boost import smoteBoost
from imba_algo.ramo import RAMOBoost

numpy.set_printoptions(threshold='nan')
categorical_index = (1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 20)
suppose_drop_cols=[5, 6, 9]
origin_data = {"train":"train.csv",
                "test":"test.csv"
              }
impute_data = {"edu":{"train":'impute_data/imputed_edu_train_str.csv',
                      "test":'impute_data/imputed_edu_test_str.csv'},
               "encode":{"train":"impute_data/encode_impute_train.csv",
                        "test":"impute_data/encode_impute_test.csv"}
              }


class MarketData(object):
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    def __init__(self, X_train, y_train, X_test, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

def basic_process(data_filename):
    with open(data_filename, 'r') as data_file:
        data_csv = csv.reader(data_file, delimiter=',')
        next(data_csv)
        data = list(data_csv)
    data = numpy.array(data)
    return data

def generate_data(sampli=False, scale=False, drop_cols=None, dt_file=origin_data):
    print("using original data...")
    train_data = basic_process(dt_file["train"])
    test_data = basic_process(dt_file["test"])

    # delete id column
    train_data = numpy.delete(train_data, 0, 1)
    test_data = numpy.delete(test_data, 0, 1)

    # One of K encoding of categorical data
    encoder = preprocessing.LabelEncoder()
    for j in (1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 20):
        train_data[:, j] = encoder.fit_transform(train_data[:, j])
        test_data[:, j] = encoder.fit_transform(test_data[:, j])

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]

    # drop feature
    if drop_cols:
        X_train = numpy.delete(X_train, drop_cols, axis=1)
        X_test = numpy.delete(X_test, drop_cols, axis=1)

    # Converting numpy strings to floats
    X_train = X_train.astype(numpy.float)
    y_train = y_train.astype(numpy.int)
    X_test = X_test.astype(numpy.float)

    # scaling
    if scale:
        X_train, X_test = scaling(X_train, X_test, axis=0)

    # sampling
    if sampli:
        X_train, y_train = sampling(X_train, y_train, strategy=sampli)

    market = MarketData(X_train, y_train, X_test)
    return market

def generate_encode_data(sampli=False, scale=True, dt_file=impute_data["encode"]):
    print("using encoded data...it is for SVM...")
    # this version data has label on train data
    train_data = numpy.loadtxt(dt_file["train"], delimiter=",", skiprows=1)
    test_data = numpy.loadtxt(dt_file["test"], delimiter=",", skiprows=1)
    # delete id column
    train_data = numpy.delete(train_data, 0, 1)
    test_data = numpy.delete(test_data, 0, 1)
    #
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    # Converting numpy strings to floats
    X_train = X_train.astype(numpy.float)
    y_train = y_train.astype(numpy.int)
    X_test = X_test.astype(numpy.float)
    print("data", X_train.shape, y_train.shape, X_test.shape)
    # scaling
    if scale:
        X_train, X_test = scaling(X_train, X_test, axis=0, norm=False)
    # sampling
    if sampli:
        X_train, y_train = sampling(X_train, y_train, strategy=sampli)
    market = MarketData(X_train,  y_train, X_test)
    return market


def sampling(X_data, y_data, strategy="combine"):
    print("%s sampling..." % strategy)
    if strategy == "under":
        clf = NearMiss(version=1, random_state=0)
    if strategy == "over-smote":
        clf = SMOTE(random_state=0)
    if strategy == "combine":
        clf = SMOTEENN(random_state=0, ratio='minority')
    X_data, y_data = clf.fit_sample(X_data, y_data)
    return X_data, y_data

def scaling(X_train, X_test, axis=0, norm=True):
    if norm:
        print("normilizing on all the input...")
        X_train = preprocessing.normalize(X_train, norm='l2')
        X_test = preprocessing.normalize(X_test, norm='l2')
        return X_train, X_test

    print("scaling on all the input...")
    X_train = preprocessing.scale(X_train, axis=0)
    X_test = preprocessing.scale(X_test, axis=0)
    return X_train, X_test

def trainModel(clf, data):
    scorer = make_scorer(matthews_corrcoef)
    clf_scores = []
    for i in range(3):
        scores = cross_val_score(clf, data.X_train, data.y_train, cv=10, scoring=scorer)
        print("cross valid mcc:", scores.mean())
        clf_scores.append(scores.mean())
    return numpy.mean(clf_scores)


def PredictTest(clf, data, result_filename="submission.csv", downfile=False, verbose=True):
    clf = clf.fit(data.X_train, data.y_train)
    train_pred = clf.predict(data.X_train)
    train_acc = recall_score(data.y_train, train_pred, average='micro')
    train_mcc = matthews_corrcoef(data.y_train, train_pred)
    y_pred = clf.predict(data.X_test)
    if verbose:
        print("train mcc:", train_mcc, "acc:", train_acc)
    else:
        return train_mcc
    if not downfile:
        return
    with open(result_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(0, len(data.X_test)):
            f.write(','.join([str(i), str(int(y_pred[i]))]))
            f.write('\n')

def decision_tree(max_depth=7, cw=None, max_features=20):
    DT = tree.DecisionTreeClassifier(max_depth=max_depth, class_weight=cw, max_features=max_features)
    return DT

def adaboost(dt, n_estimators=100, lr=1):
    bdt = AdaBoostClassifier(dt, algorithm="SAMME.R",
                             n_estimators=n_estimators, learning_rate=lr)
    return bdt

def svm_clf(kernel='linear',cw='balanced', gamma='auto'):
    clf = svm.SVC(C=1, kernel=kernel, class_weight=cw, gamma=gamma,cache_size=1000)
    return clf

def nu_svm(cw="balanced"):
    clf = svm.NuSVC(cache_size=1000, class_weight=cw)
    return clf

def gbc(nest=200, dep=1, lr=1, mf=18):
    clf = GradientBoostingClassifier(n_estimators=nest, learning_rate=lr,
                                     max_depth=dep, random_state=0, max_features=mf)
    return clf

def call_smoteboost(data, nest=200, lr=0.1, dep=3):
    dt = decision_tree(max_depth=dep, cw="balanced")
    lr = 0.1
    clf = smoteBoost(base_estimator=dt, n_estimators=nest, learning_rate=lr)
    print(clf)
    trainModel(clf, data)
    res_file = "submission/smote/nest_%d_lr_%f" % (nest, lr)
    clf = clf.fit(data.X_train, data.y_train, minority_target=1)
    train_pred = clf.predict(data.X_train)
    train_acc = recall_score(data.y_train, train_pred, average='micro')
    train_mcc = matthews_corrcoef(data.y_train, train_pred)
    print("train mcc:", train_mcc, "acc:", train_acc)
    y_pred = clf.predict(data.X_test)
    with open(res_file, 'w') as f:
        f.write('id,prediction\n')
        for i in range(0, len(data.X_test)):
            f.write(','.join([str(i), str(int(y_pred[i]))]))
            f.write('\n')

def call_ramoboost(data, nest=200, lr=0.1, dep=3):
    dt = decision_tree(max_depth=dep, cw="balanced")
    clf = RAMOBoost(base_estimator=dt, n_estimators=nest, learning_rate=lr)
    print(clf)
    trainModel(clf, data)
    res_file = "submission/smote/nest_%d_lr_%f" % (nest, lr)
    clf = clf.fit(data.X_train, data.y_train, minority_target=1)
    train_pred = clf.predict(data.X_train)
    train_acc = recall_score(data.y_train, train_pred, average='micro')
    train_mcc = matthews_corrcoef(data.y_train, train_pred)
    print("train mcc:", train_mcc, "acc:", train_acc)
    y_pred = clf.predict(data.X_test)
    with open(res_file, 'w') as f:
        f.write('id,prediction\n')
        for i in range(0, len(data.X_test)):
            f.write(','.join([str(i), str(int(y_pred[i]))]))
            f.write('\n')

if __name__ == '__main__':
    market_data = generate_encode_data(sampli="combine", scale=True)
    clf = nu_svm
    trainModel(clf, market_data)
    PredictTest(clf, market_data, downfile=True)
