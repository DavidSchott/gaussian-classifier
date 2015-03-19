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


#Returns lowest distance of one vector and a feature matrix
def vecDist(v1,f2):
    min_dist = float('inf')
    for v in f2:
        min_dist = min(eucdist(v,v1),min_dist)
    return min_dist

# Returns tuple consisting of (closest distance, index of corresponding feature matrix)
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
            v3 = [min_dist, i]
    return v3

"""Returns an array of size 5000 consisting of [vector from features matrix, corresponding classification]
   Parameters: f1 = feature-vector we wish to classify
               fs = other training feature-vectors, arranged in increasing order starting from lowest fold-number
               f1Index = the fold-number of f1."""
def computeFeatClasses(f1,f1Index):
    vArr = [([],0)]
    vecNo = 0
    for v in f1:
        print("Vector "+vecNo + " of fold_class: "+f1Index) #used for testing
        min_dist = float('inf')
        foldNo = 0

        while (foldNo < 10):
            #Ensure correct foldNo
            foldNo = foldNo + 1
            if (foldNo == f1Index):
                foldNo = foldNo + 1

            #Access correct dict-entry
            f = data.get("fold"+str(foldNo)+"_features")
            #Compute lowest distance between vector and entire feature-vector, return distance and feature-vector.
            temp_dist = closestVec(v,f)

            if (temp_dist[0] <= min_dist):
                min_dist = temp_dist[0]
                class_entry = "fold"+ str(foldNo) + "_classes"
                fold_class = data.get(class_entry)[temp_dist[1]][0]
                vArr.append((v,fold_class))
        vecNo = vecNo + 1 # used for testing
    return vArr

""" Returns an dictionary with key "fold'No'_features", corr. item = array of size 5000 [vector of fold'No'_features, classification number]"""
def knn1():
    foldNo = 0
    #Dict with keys "fold'No'_features", items = array [vector of fold'No'_features, classification number]
    vClass_dict = {"",[]}
    while (foldNo < 10):
        foldNo = foldNo + 1
        foldname = "fold" + str(foldNo)+"_features"
        f = data.get(foldname)
        vClass_dict[foldname] = computeFeatClasses(f,foldNo)

    return vClass_dict



#Used for testing and debugging
def main():
    v1 = data.get("fold1_features")[1]
    f2 = data.get("fold2_features")

    min = computeFeatClasses(f2,2)
    return (min)
