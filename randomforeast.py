import numpy as np
import pandas as pd
import heapq
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier as rfc

data = pd.read_table("zipcombo.dat", sep="\s+")
#data = pd.read_table("dtrain123.dat", sep="\s+")
data = np.array(data)

def RFC(test_data, test_label, train_data, train_label, d):
    rfc_classifier = rfc(n_estimators = d)
    #It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
    rfc_train_error = rfc_classifier.fit(train_data, train_label).score(train_data, train_label)
    rfc_test_error= rfc_classifier.score(test_data, test_label)
    y_predict = rfc_classifier.predict(test_data)
    return y_predict, (1-rfc_train_error)*len(train_data), (1-rfc_test_error)*len(test_data)

def test():
    print("ssssss")
    data_train ,data_test = train_test_split(data,test_size=0.2)
    x_train = data_train[:,1:]
    y_train = data_train[:,0]
    x_test = data_test[:,1:]
    y_test = data_test[:,0]
    pred, train_error, test_error = RFC(x_test, y_test, x_train, y_train, 500)
    print(pred)
    for i in range(len(pred)):
        print(str(pred[i]) + " : " + str(y_test[i]))
    print(train_error)
    print(test_error)
if __name__ == '__main__':
    test()