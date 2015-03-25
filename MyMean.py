__author__ = 's1329380'
#import scipy.io, math, pprint
import numpy as np
import knn

#Sum of columns / no of rows
def mean(m1):
    sum_cols = m1.sum(axis = 0)
    mean_cols = sum_cols
    for sum in sum_cols:
        mean_cols = sum / len(m1)
    return mean_cols

def mean(vec_feat):
    columnSum = 0
    vec_mean = []
    for i in range(len(vec_feat[0])):
        for j in range(len(vec_feat)):
            columnSum = columnSum + vec_feat[j][i]
        vec_mean.append(columnSum)
        columnSum = 0
    for element in range(len(vec_mean)):
        vec_mean[element] = vec_mean[element] / (len(vec_feat) + 0.0)
    return np.array(vec_mean)


def MyCov(feats, meanvec):
    cov = [[0 for i in range(len(feats[0]))] for j in range(len(feats[0]))]
    for i in range(len(feats)):
        diff = np.asmatrix(feats[i]) - np.asmatrix(meanvec)
        diffT = diff.transpose()
        cov = cov + (diffT * diff)
    return cov * (1.0/(len(feats)-1.0))


"""Returns an dictionary with keys "classNo", containing a list of all vectors belonging to that foldNo-class"""
def classDict():
    dict = {}
    for cl_no in range(1, 11):
        list_temp = []
        for j in range(1, 11):
            key = "fold"+str(j)
            vectors = knn.data[key+"_features"]    # this is one fold_features with 5000 vectors
            real_class = knn.data[key+"_classes"]  # this is the corresponding fold_class
            for i in range(len(vectors)):
                if (real_class[i][0] == cl_no):
                    list_temp.append(vectors[i])
        dict[cl_no] = (list_temp)               #a list of lists of vectors
    return dict

def MyMean(classdict):
    means = {}
    for key in classdict:
        means[key+"_mean"] = mean(classdict[key])
    return means


#Used for testing
def main():
    m1 = np.asmatrix([[4.,2.,0.6],
                     [4.2,2.1,0.59],
                     [3.9,2.,0.58],
                     [4.3,2.1,0.62],
                     [4.1,2.2,0.63]])
    print(mean(m1))
    return classDict()
