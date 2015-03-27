__author__ = 's1329380'
#import scipy.io, math, pprint
import numpy as np
import knn
import scipy as sp

#Sum of columns / no of rows. Only works for np-Matrix.
def meanMatrix(m1):
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
