__author__ = 's1329380'
import scipy.io
import numpy as np
from scipy.spatial.distance import cdist as fastdist # finds distances of all matrices
import math, time

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

feats = getfeats() #contains all feature vectors
classes = getclasses() #contains classes

# Constructor
def __init__(self):
    self.data = scipy.io.loadmat("cifar10.mat")

# Methods for OLD knn-classification:
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
    v3 = ([],0)
    i = 0
    for v in f2:
        i = i+1
        temp_dist = eucdist(v,v1)
        if (temp_dist <= min_dist):
            min_dist = temp_dist
            v3 = [min_dist, i]
    return v3


"""Improved knn-classification using cdist"""

""" Computes and stores all distances in the 5000x5000 matrix [
 [dist(f1_vec1, f2_vec1), dist(f1_vec1, f2_vec2),.., dist(f1_vec1, f2_vec5000)],
 [dist(f1_vec2, f2_vec1), ...]
 [dist(f1_vec5000, f2_vec1), dist(f1_vec5000, f2_vec2), ... , dist(f1_vec5000, f2_vec5000)] ]
 for each feature_class different to fold"f1foldNo"_features in a dict."""
def getAllDistances(f1,f1foldNo):
    foldNo = 9
    distdict = {} #f1foldNo = 10
    while (foldNo < 10):
        foldNo = foldNo + 1
        if(foldNo == f1foldNo and f1foldNo == 10): #case for f1foldno = 10
            break
        if (foldNo == f1foldNo):
            foldNo = foldNo + 1
        fkey = "fold"+str(foldNo)+"_features"
        f = data.get(fkey)
        f1_fdists = fastdist(f1,f)
        distdict[fkey] = f1_fdists
        print("Completed computing distances of: "+ fkey) # used for testing
    return distdict

"""Returns an array of tuples of the form (vector of f1, estimated fold_class of f1)"""
def computeFoldClasses1(f1,f1foldNo):
    dists_dict = getAllDistances(f1,f1foldNo)
    vArr = np.empty(5000, dtype=object) #store closest vector for each vector in f1.
    f1vec_index = 0

    while (f1vec_index < 5000):
        #initialize/reset all necessary variables for closest vector
        min_dist = float('inf')
        fvecNo_perm = 0
        foldNo = 0
        foldNo_perm = 0
        while (foldNo < 10):
            foldNo = foldNo + 1
            if(foldNo == f1foldNo and f1foldNo == 10):
                break
            if (foldNo == f1foldNo):
                foldNo = foldNo + 1

            #Gather dict for current fold
            fkey = "fold"+str(foldNo)+"_features"
            f_dists = dists_dict[fkey]
            fvec_index = 0
            """iterate through dict to find closest point"""
            for dist in f_dists[f1vec_index]: #for vector 0
                if (dist <= min_dist):
                    min_dist = min(dist,min_dist)
                    fvecNo_perm = fvec_index
                    foldNo_perm = foldNo
                fvec_index = fvec_index + 1
        vArr[f1vec_index] = (f1[f1vec_index],classes[foldNo_perm-1][fvecNo_perm])
        #(min_dist, fvecNo_perm, classes[foldNo_perm-1][fvecNo_perm][0]) # class_entry = "fold"+ str(foldNo) + "_classes", class_entry[fvec_index][0]
                                     #str(fvecNo_perm)+ "'th vector in fold_"+str(foldNo_perm)+"features") #closest vector of f2 to f1[f1vec_index]
        f1vec_index = f1vec_index + 1
        print("Vector "+str(f1vec_index) + " completed of fold_class: "+ str(f1foldNo)) #used for testing
    return vArr


""" Returns an dictionary with key "fold'No'_features", corr. item = array of size 5000 [vector of fold'No'_features, classification number]"""
def knn1():
    foldNo = 0
    #Dict with keys "fold'No'_features" : items = array [given vector of fold'No'_features, corresponding classification number]
    vClass_dict = {"" : []}
    while (foldNo < 10):
        foldNo = foldNo + 1
        foldname = "fold" + str(foldNo)+"_features"
        f = data.get(foldname)
        start = time.time()
        vClass_dict[foldname] = computeFoldClasses1(f,foldNo)
        end = time.time()
        print("completed in time:")
        print(end - start)
    return vClass_dict


def computeFoldClassesN(f1, f1foldNo,N):
    dists_dict = getAllDistances(f1,f1foldNo)
    vArr = np.empty(5000, dtype=object) #store closest vector for each vector in f1.
    f1vec_index = 0

    while (f1vec_index < 5000):
        #initialize/reset all necessary variables for closest vector
        min_dist = float('inf')
        fvecNo_perm = 0
        foldNo = 0
        foldNo_perm = 0
        while (foldNo < 10):
            foldNo = foldNo + 1
            if (foldNo == f1foldNo):
                foldNo = foldNo + 1

            #Gather dict for current fold
            fkey = "fold"+str(foldNo)+"_features"
            f_dists = dists_dict[fkey]
            fvec_index = 0
            """iterate through dict to find closest N points"""
            for dist in f_dists[f1vec_index]: #for vector 0
                if (dist <= min_dist):
                    min_dist = min(dist,min_dist)
                    fvecNo_perm = fvec_index
                    foldNo_perm = foldNo
                fvec_index = fvec_index + 1
        vArr[f1vec_index] = (f1[f1vec_index],classes[foldNo_perm-1][fvecNo_perm])
        f1vec_index = f1vec_index + 1
        print("Vector "+str(f1vec_index) + " completed of fold_class: "+ str(f1foldNo)) #used for testing
    return vArr

 #(min_dist, fvecNo_perm, classes[foldNo_perm-1][fvecNo_perm][0]) # class_entry = "fold"+ str(foldNo) + "_classes", class_entry[fvec_index][0]
                                     #str(fvecNo_perm)+ "'th vector in fold_"+str(foldNo_perm)+"features") #closest vector of f2 to f1[f1vec_index]



"""Used for testing and debugging"""
f1 = data.get("fold1_features")
f10 = data.get("fold10_features")
def main():
    m1 = np.asmatrix([[4.,2.,0.6],
                     [4.2,2.1,0.59],
                     [3.9,2.,0.58],
                     [4.3,2.1,0.62],
                     [4.1,2.2,0.63]])

    m2 = np.asmatrix([[4.,2.,0.6],
                     [1,1,1],
                     [0,0,0],
                     [0,0,0],
                     [4.1,2.2,0.63]])

    return getAllDistances(f10,10)



# Redundant code
"""
Returns closest vectors of f1 feature matrix with corresponding fold-class.
def closestDistAll(f1,f1foldNo):
    min_dist = float('inf')
    foldNo = 0
    vArr = []
    dists_arr = []
    while (foldNo < 10):
        foldNo = foldNo + 1

        if (foldNo == f1foldNo):
            foldNo = foldNo + 1

        f = data.get("fold"+str(foldNo)+"_features")
        #Compute lowest distance between vector and entire feature-vector, return distance and feature-vector.
        temp_tuple_arr = closestDist(f1,f,foldNo)
        dists_arr.append(temp_tuple_arr)

        for temp_tuple in temp_tuple_arr:
            if (temp_tuple[0] <= min_dist):
                min_dist = temp_tuple[0]
                class_entry = "fold"+ str(foldNo) + "_classes"
                fold_class = data.get(class_entry)[temp_tuple[1]][0]
                vArr.append((temp_tuple[2], fold_class))
    return vArr"""
# Returns array consisting of tuples(closest distance, index of corresponding feature matrix vector)
def closestDist(f1,f2,f2_foldNo):
    """Computes and stores all distances in the 5000x5000 matrix [ [dist(f1_vec1, f2_vec1), dist(f1_vec1, f2_vec2),.., dist(f1_vec1, f2_vec5000)],
                                                                 [dist(f1_vec2, f2_vec1), ...]
                                                                 [dist(f1_vec5000, f2_vec1), dist(f1_vec5000, f2_vec2), ... , dist(f1_vec5000, f2_vec5000)] ]"""
    dists = fastdist(f1,f2)
    vArr = np.empty(5000, dtype=object) #store closest vector of f2 for each vector in f1.
    f1vecNo = 0
    while (f1vecNo < 5000):
        j = 0
        min_dist = float('inf')
        for dist in dists[f1vecNo]:

            if (dist <= min_dist):
                min_dist = dist
                vArr[f1vecNo] = (min_dist, j,feats[f2_foldNo-1][j])
                                 #feats[f2_foldNo][j],j) #str(j)+ "'th vector in fold_"+str(f2_foldNo)+"features") #closest vector of f2 to f1[f1vecNo]
            j = j + 1
        print(str(f1vecNo) + " completed of fold_features" + str(f2_foldNo))
        f1vecNo = f1vecNo + 1
    return vArr
"""Returns an array of size 5000 consisting of [vector from features matrix, corresponding classification]
   Parameters: f1 = feature-vector we wish to classify
               f1Index = the fold-number of f1."""
def computeFoldClassesOld(f1,f1Index):
    vArr = [([],0)]
    vecNo = 0
    for v in f1:
        print("Vector "+str(vecNo) + " of fold_class: "+ str(f1Index)) #used for testing
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
def knn1Slow():
    foldNo = 0
    #Dict with keys "fold'No'_features", items = array [vector of fold'No'_features, classification number]
    vClass_dict = {"" : []}
    while (foldNo < 10):
        foldNo = foldNo + 1
        foldname = "fold" + str(foldNo)+"_features"
        f = data.get(foldname)
        vClass_dict[foldname] = computeFoldClassesOld(f,foldNo)

    return vClass_dict