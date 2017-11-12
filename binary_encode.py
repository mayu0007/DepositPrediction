import csv

import numpy
from sklearn import preprocessing
from sklearn import tree, svm
from sklearn.metrics import make_scorer, matthews_corrcoef, recall_score
from sklearn.model_selection import cross_val_score
import category_encoders as ce

numpy.set_printoptions(threshold='nan')
categorical_index = (1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 20)


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

def basic_process(data_filename, delimiter=","):
    with open(data_filename, 'r') as data_file:
        data_csv = csv.reader(data_file, delimiter=delimiter)
        next(data_csv)
        data = list(data_csv)
    data = numpy.array(data)
    return data

def cat_encode(X, y, encode_type=None):
    out = None
    if encode_type == "label":
        encoder = preprocessing.LabelEncoder()
        for j in (1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 20):
            X[:, j] = encoder.fit_transform(X[:, j])
        out = X
    else:
        encoder = ce.BinaryEncoder(cols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 21])
        encoder.fit(X, y)
        out = encoder.transform(X)
        print(out.info())
    print("encode out type", type(out))
    return out

def generate_data(sampli=False, scale=False):
    train_data = basic_process('train.csv')
    test_data = basic_process('test.csv')
    full_data = numpy.concatenate((train_data, test_data), axis=0)
    #print("train shape:", train_data.shape, "test shape:", test_data.shape, "full_data:", full_data.shape)
    fix_col=numpy.zeros((full_data.shape[0], 1))
    full_data = numpy.concatenate((fix_col, full_data), axis=1)

    l_encoder = preprocessing.LabelEncoder()
    train_data[:, 21] = l_encoder.fit_transform(train_data[:, 21])

    encoder = ce.BinaryEncoder(cols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 15])
    encoder.fit(full_data)
    new_train_data = encoder.transform(train_data)
    print(new_train_data.info())
    new_train_data.to_csv("impute_data/encode_impute_train.csv", sep=',')
    new_test_data = encoder.transform(test_data)
    print(new_test_data.info())
    new_test_data.to_csv("impute_data/encode_impute_test.csv", sep=',')


if __name__ == '__main__':
    market_data = generate_data(sampli=True, scale=False)
