__author__ = 's1329380'
import scipy.io
import numpy as np
import math


#Dict containing data
data = scipy.io.loadmat("cifar10.mat")

#Returns all fold_features, listed in order 1..10. Thus return array of the form [feature vector[vector],.. ]
def getfeats():
    foldNo = 0
    feat_arr = []
    while (foldNo < 10):
        foldNo = foldNo + 1
        f = (data.get("fold"+str(foldNo)+"_features"))
        feat_arr.append(f)
    return np.array(feat_arr)

def getclasses():
    foldNo = 0
    class_arr = []
    while (foldNo < 10):
        foldNo = foldNo + 1
        f = (data.get("fold"+str(foldNo)+"_classes"))
        class_arr.append(f)
    return np.array(class_arr)


''' def __init__(self):
        data = scipy.io.loadmat("cifar10.mat")
        feats = self.getfeats()'''


feats = getfeats() #contains all feature vectors
classes = getclasses() #contains classes



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
    #vArr = [([], 0)]
    vArr = []
    vecNo = 0
    f1Index = f1Index-1
    for v in f1:
        print("Vector "+str(vecNo) + " of fold_class: "+ str(f1Index)) #used for testing
        min_dist = float('inf')
        foldNo = 0
        while (foldNo < 10):
            if (foldNo == f1Index):
                foldNo = foldNo + 1
                #Access correct dict-entry
            f = feats[foldNo]
            #Ensure correct foldNo
            foldNo = foldNo + 1

            #Compute lowest distance between vector and entire feature-vector, return distance and feature-vector.
            temp_dist = closestVec(v, f)

            if (temp_dist[0] <= min_dist):
                min_dist = temp_dist[0]
                class_entry = classes[foldNo-1]
                fold_class = (class_entry)[temp_dist[1]][0]
                vArr.append((v,fold_class))
        vecNo = vecNo + 1 # used for testing
    return np.array(vArr)


""" Returns an dictionary with key "fold'No'_features", corr. item = array of size 5000 [fold'No'_features, predicted classification number]"""
def knn1():
    foldNo = 0
    #Dict with keys "fold'No'_features", items = array [vector of fold'No'_features, classification number]
    vClass_dict = {"" : []}
    while (foldNo < 10):
        foldNo = foldNo + 1
        foldname = "fold" + str(foldNo)+"_features"
        f = feats[foldNo-1]
        vClass_dict[foldname] = computeFeatClasses(f, foldNo)

    return vClass_dict



#Used for testing and debugging
def main():
    f2 = data.get("fold2_features")
    #return feats
    min = computeFeatClasses(f2,2)
    return (min)
