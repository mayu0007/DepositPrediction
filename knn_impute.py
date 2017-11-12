#
# credit to Li Xinyu
#
import csv
import numpy
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

numpy.set_printoptions(threshold='nan')
attr_index = {'id':0,'age': 1, 'job': 2, 'marital': 3, 'education': 4, 'default': 5,
         'housing': 6, 'loan': 7, 'contact': 8, 'month': 9, 'day_of_week': 10,
         'duration': 11, 'campaign': 12, 'pdays': 13, 'previous': 14,
         'poutcome': 15, 'emp.var.rate': 16, 'cons.price.idx': 17,
         'cons.conf.idx': 18, 'euribor3m': 19, 'nr.employed': 20, 'y': 21}
MissValue = {"education": 'unknown', "housing": 'unknown'}
categorical_index = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 21]
header = ['id', 'age', 'job', 'marital', 'education', 'default', 'housing',
          'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign',
          'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
          'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']
best_ns = {'education': 39, 'housing': 45}
attr_classes = {"education":['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                             'illiterate', 'professional.course',
                             'university.degree', 'unknown'],
                "housing":['no', 'unknown', 'yes']}



class MarketData(object):
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


# use knn for impute missing values
def knn(X_train, y_train, plot=True, best_n=None):
    if best_n:
        # prediction
        clf = KNeighborsClassifier(n_neighbors=best_n)
        clf.fit(data.X_train, data.y_train)
        y_pred = clf.predict(data.X_test)
        return y_pred, clf
    knn_scores = []
    for n_neighbors in range(4, 51):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        knn_scores.append((n_neighbors, scores.mean()))
    if plot:
        X = list(range(4, 51))
        Y = [s[1] for s in knn_scores]
        plt.plot(X, Y)
        plt.xlim(0, 50)
        plt.xlabel("n_neighbors")
        plt.ylabel("housing score")
        plt.show()
    knn_scores = sorted(knn_scores, key=lambda x: x[1], reverse=True)
    print("top 5 knn:", knn_scores[:5])

def train_model(X_train, y_train, best_n=5):
    clf = KNeighborsClassifier(n_neighbors=best_n)
    clf.fit(X_train, y_train)
    return clf

def knn_predict(model, impute_X):
    imputed_pred = model.predict(impute_X)
    return imputed_pred

def fixin(fix_data, fix_mask, nan_col, impute_pred):
    fix_data[fix_mask, nan_col] = impute_pred
    return fix_data

def knn_impute(model, t_data, nan_col, nan_value):
    impute_data = t_data[t_data[:, nan_col] == nan_value]
    impute_mask = list(impute_data[:, 0].astype(numpy.int))
    #delete missing col
    impute_data = numpy.delete(impute_data, nan_col, 1)
    impute_data = numpy.delete(impute_data, -1, 1) # del label col
    impute_data = numpy.delete(impute_data, 0, 1) # del id col

    impute_pred = knn_predict(model, impute_data)
    t_data = fixin(t_data, impute_mask, nan_col, impute_pred)
    return t_data, impute_pred, impute_mask

def basic_process(data_filename, delimiter=','):
    with open(data_filename, 'r') as data_file:
        data_csv = csv.reader(data_file, delimiter=delimiter)
        next(data_csv)
        data = list(data_csv)
    np_data = numpy.array(data)
    print("in basic:", list(np_data[0]))
    return np_data

def label_process(data, nan_attr):
    nan_col = attr_index[nan_attr]
    nan_value = 0
    # One of K encoding of categorical data
    encoder = preprocessing.LabelEncoder()
    for j in categorical_index:
        data[:, j] = encoder.fit_transform(data[:, j])
        if j == nan_col:
            classes = list(encoder.classes_)
            print(classes)
            nan_value = classes.index(MissValue[nan_attr])
            print(nan_value)
    data = data.astype(numpy.float)
    return data, nan_col, nan_value

def recover_data(origin_data, impute_pred, impute_mask, nan_col, nan_attr):
    nan_classes = attr_classes[nan_attr]
    impute_value = [nan_classes[int(y)] for y in impute_pred]
    print("recover impute value",type(impute_value), "origin_data",type(origin_data))
    origin_data[impute_mask, nan_col] = impute_value
    #recover_data = numpy.insert(origin_data, 0, header, axis=0)
    print("recover data:", origin_data.shape)
    return origin_data

if __name__ == '__main__':
    nan_info = [("education", 4, 7), ("housing", 6, 1)]

    train_data = basic_process('train.csv')  # read data from csv
    o_train_data = train_data.copy()

    train_data, _, _ = label_process(train_data.copy(), "housing")

    test_data = basic_process('test.csv')  # read data from csv
    o_test_data = test_data.copy()
    test_data, _, _ = label_process(test_data, "education")

    full_data = numpy.concatenate((train_data, test_data), axis=0)
    full_data = full_data[:, :-1]
    print("full data:", full_data.shape)

    for nan_attr, nan_col, nan_value in nan_info:
        full_train = full_data[full_data[:, nan_col] != nan_value]
        print("full train:", full_train.shape, nan_attr)
        y_train = full_train[:, nan_col]
        X_train = numpy.delete(full_train, nan_col, 1)
        X_train = numpy.delete(X_train, 0, 1) #del id col
        if nan_attr == "douniwan":
            knn(X_train, y_train)

        model = train_model(X_train, y_train, best_ns[nan_attr])
        test_data, test_pred, test_mask = knn_impute(model, test_data, nan_col, nan_value)
        train_data, train_pred, train_mask = knn_impute(model, train_data, nan_col, nan_value)

        o_train_data = recover_data(o_train_data, train_pred, train_mask, nan_col, nan_attr)
        o_test_data = recover_data(o_test_data, test_pred, test_mask, nan_col, nan_attr)

    numpy.savetxt("impute_data/imputed_test.csv", test_data, delimiter=",")
    numpy.savetxt("impute_data/imputed_train.csv", train_data, delimiter=",")


    recover_train_data = numpy.insert(o_train_data, 0, header, axis=0)
    recover_test_data = numpy.insert(o_test_data, 0, header, axis=0)
    numpy.savetxt("impute_data/imputed_edu_test_str.csv", recover_test_data, delimiter=",", fmt="%s")
    numpy.savetxt("impute_data/imputed_edu_train_str.csv", recover_train_data, delimiter=",", fmt="%s")
