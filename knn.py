__author__ = 's1329380'
from pprint import pprint
import scipy.io
#from numpy import *
import numpy as np
import math

#Dict containing data
data = scipy.io.loadmat("cifar10.mat")

# Constructor
def __init__(self):
    self.data = scipy.io.loadmat("cifar10.mat")

# Methods for knn-classification:
#Norm
def myNorm(v1):
    sum = 0.0
    for no in v1:
        sum += math.pow(no,2)
    return math.pow(sum,0.5)


#Euclidean distance
def eucdist(v1,v2):
    v3 = np.subtract(v1,v2)
    return myNorm(v3)


#Returns lowest distance of one vector and a vectors of feature vector
def vecDist(v1,f2):
    min_dist = float('inf')
    for v in f2:
        min_dist = min(eucdist(v,v1),min_dist)
    return min_dist

# Returns tuple consisting of (closestvector, index of feature vector)
def closestVec(v1,f2):
    min_dist = float('inf')
    temp_dist = 0.0
    v3 = ([],0)
    i = 0
    for v in f2:
        i = i+1
        temp_dist = eucdist(v,v1)
        if (temp_dist <= min_dist):
            min_dist = temp_dist
            v3 = [v, i]
    return v3

"""Returns an array of size 5000 consisting of [closest-vector to each v, corresponding index, corresponding fold-class description]
   Parameters: f1 = feature-vector we wish to classify
               fs = other training feature-vectors, arranged in increasing order starting from lowest fold-number
               f1Index = the fold-number of f1."""
#WRONG
def computeFeatClasses(f1,fs,f1Index):
    vArr = [[],0,""]
    vecNo = 0
    for v in f1:
        min_dist = float('inf')
        featNo = 0

        for f in fs:
            featNo = featNo + 1
            if (featNo == f1Index):
                featNo = featNo + 1

            temp_dist = vecDist(v,f)

            if (temp_dist <= min_dist):
                min_dist = temp_dist
                featClass = "fold"+ str(featNo) + "_class"
                vArr.append(v,vecNo,featClass)

        vecNo = vecNo + 1

    return vArr




#Used for testing and debugging
def main():
    v1 = data.get("fold1_features")[1]
    f2 = data.get("fold2_features")

    min = closestVec(v1,f2)
    return (min)