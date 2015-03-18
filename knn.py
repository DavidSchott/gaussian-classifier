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

# Returns closest vector between vector and feature-vector
def closestVec(v1,f2):
    min_dist = float('inf')
    v3 = []
    for v in f2:
        if (eucdist(v,v1) <= min_dist):
            v3 = v
            min_dist = min(eucdist(v,v1),min_dist)
    return v3

def closestVecFeat(f1,f2):
    for v in f1:
        closestVec(v,f2)

#def myKnn():

#Used for testing and debugging
def main():
    v1 = data.get("fold1_features")[1]
    f2 = data.get("fold2_features")

    min = closestVec(v1,f2)
    print(min)