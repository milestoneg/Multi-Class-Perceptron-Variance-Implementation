import numpy as np
import pandas as pd
import heapq
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_table("zipcombo.dat", sep="\s+")
#data = pd.read_table("dtrain123.dat", sep="\s+")
data = np.array(data)

def cos_knn(k, test_data,  train_data, train_label):
   
    # find cosine similarity for every point in test_data between every other point in stored_data
    cos_similarity = cosine_similarity(test_data, train_data)
    
    # get top k indices of images in stored_data that are most similar to any given test_data point
    top = [(heapq.nlargest((k), range(len(i)), i.take)) for i in cos_similarity]
    
    # convert indices to numbers using stored target values
    top = [[train_label[j] for j in i[:k]] for i in top]
    
    # vote, and return prediction for every image in test_data
    pred = [max(set(i), key=i.count) for i in top]
    pred = np.array(pred)
    return pred

def test():
    print("ssssss")
    data_train ,data_test = train_test_split(data,test_size=0.2)
    x_train = data_train[:,1:]
    y_train = data_train[:,0]
    x_test = data_test[:,1:]
    y_test = data_test[:,0]
    pred = cos_knn(10, x_test, x_train, y_train)
    print(pred)
    for i in range(len(pred)):
        print(str(pred[i]) + " : " + str(y_test[i]))

if __name__ == '__main__':
    test()