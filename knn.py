__author__ = 's1329380'
import scipy.io
import numpy as np
from scipy.spatial.distance import cdist as fastdist # finds distances of all matrices
import time

#Dict containing data
data = scipy.io.loadmat("cifar10.mat")
#Arrays containing predicted class
knn1_dict = {}
knn3_dict = {}
knn5_dict = {}

knn1_conf = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
knn3_conf = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
knn5_conf = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
knn1_acc = 0.0
knn3_acc = 0.0
knn5_acc = 0.0

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

"""Improved knn-classification using cdist"""
""" Computes and stores all distances in the  9x5000x5000 matrix [
 [dist(f1_vec1, f2_vec1), dist(f1_vec1, f2_vec2),.., dist(f1_vec1, f2_vec5000)],            = 5000
 [dist(f1_vec2, f2_vec1), ... dist(f1_vec2, f2_vec5000)],...,                               = 2x5000
 [dist(f1_vec5000, f2_vec1), dist(f1_vec5000, f2_vec2), ... , dist(f1_vec5000, f2_vec5000)] = 5000x5000
 ...
 [dist(f1_vec1, f10_vec1), dist(f1_vec1, f10_vec5000)]                                      = 9x5000x5000
 ]
 for each feature_class different to fold"f1foldNo"_features in a dict."""
def getAllDistances(f1,f1foldNo):
    foldNo = 0
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
    return distdict

#Returns the total classification accuracy for a given confusion matrix
def predAcc(conf_matrix):
    sum = 0.0
    for i in range(len(conf_matrix)):
        sum += conf_matrix[i][i]
    return sum / 50000.0

"""Returns an 5000 array of tuples of the form (vector of f1, estimated fold_class of corresponding vector in f1)"""
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
        f1vec_index = f1vec_index + 1
    return vArr

""" Returns an dictionary with key "fold'No'_features", corr. item = array of size 5000 [vector of fold'No'_features, classification number]"""
def knn1():
    foldNo = 0
    #Dict with keys "fold'No'_features" : items = array [given vector of fold'No'_features, corresponding classification number]
    vClass_dict = {}
    while (foldNo < 10):
        foldNo = foldNo + 1
        foldname = "fold" + str(foldNo)+"_features"
        f = data.get(foldname)
        start = time.time()
        vClass_dict[foldname] = computeFoldClasses1(f,foldNo)
        end = time.time()
        print(end - start)
    return vClass_dict

"""Takes the dictionary returned from knn1() and builds a confusion matrix."""
def knn1ConfMatrix(dict):
    # Rows = class i, 1-10
    # columns = classified as class j
    matrix = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
    for key in dict.keys():
        i = 0
        for tuple in dict[key]: #tuple = dict["fold1_features"][0] , real_class = data
            classskey = key[:-8] + "classes"
            real_class = data[classskey][i][0]
            predicted_class = tuple[1][0]
            matrix[real_class-1][predicted_class-1] += 1
            i +=1
    return matrix

"""KNN-5 CLASSIFICATION METHODS"""
"""Returns an 5000 array of list of the form (estimated fold_class of corresponding vector in f1)"""
def computeFoldClasses5(f1,f1foldNo):
    dists_dict = getAllDistances(f1,f1foldNo)
    vArr = np.empty(5000, dtype=object) #store closest vector for each vector in f1.
    f1vec_index = 0
    while (f1vec_index < 5000):
        #initialize/reset all necessary variables for closest vector
        min_dist = float('inf')
        l1 = [min_dist, 0]  # (distance of point, predicted class)
        l2 = l1
        l3 = l1
        l4 = l1
        l5 = l1
        foldNo = 0
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
            """iterate through dict to find closest 5 points"""
            for dist in f_dists[f1vec_index]: #for vector 0
                if (dist <= l1[0]):
                    l1[0] = dist
                    l1[1] = classes[foldNo-1][fvec_index]
                elif (dist > l1[0] and dist < l2[0]):
                    l2[0] = dist
                    l2[1] = classes[foldNo-1][fvec_index]

                elif (dist > l2[0] and dist < l3[0]):
                    l3[0] = dist
                    l3[1] = classes[foldNo-1][fvec_index]
                elif (dist > l3[0] and dist < l4[0]):
                    l4[0] = dist
                    l4[1] = classes[foldNo-1][fvec_index]
                elif (dist > l4[0] and dist < l5[0]):
                    l5[0] = dist
                    l5[1] = classes[foldNo-1][fvec_index]
                fvec_index = fvec_index + 1
        vArr[f1vec_index] = most_frequent_class([l1,l2,l3,l4,l5])
        f1vec_index = f1vec_index + 1
    return vArr


# Returns Dict with keys "fold'No'_features" : items = tuple ([class of closest point], [class of 2nd closest point]...)
def knn5():
    foldNo = 0
    vClass_dict = {}
    while (foldNo < 10):
        foldNo = foldNo + 1
        foldname = "fold" + str(foldNo)+"_features"
        f = data.get(foldname)
        start = time.time()
        vClass_dict[foldname] = computeFoldClasses5(f,foldNo)
        end = time.time()
        print("completed "+str(foldNo)+" in time:")
        print(end - start)
    return vClass_dict

"""Takes the dictionary returned from knn1() and builds a confusion matrix."""
def knn5ConfMatrix(dict):
    # Rows = class i, 1-10
    # columns = classified as class j
    matrix = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
    for key in dict.keys():
        i = 0
        for class_no in dict[key]: #class_no = dict["fold1_features"][i]
            classskey = key[:-8] + "classes"
            real_class = data[classskey][i][0] #real class = actual class
            predicted_class = class_no
            matrix[real_class-1][predicted_class-1] += 1
            i +=1
    return matrix

"""KNN-3 CLASSIFICATION METHODS"""
"""Returns an 5000 array of list of the form (estimated fold_class of corresponding vector in f1)"""
def computeFoldClasses3(f1,f1foldNo):
    dists_dict = getAllDistances(f1,f1foldNo)
    vArr = np.empty(5000, dtype=object) #store closest vector for each vector in f1.
    f1vec_index = 0
    while (f1vec_index < 5000):
        #initialize/reset all necessary variables for closest vector
        min_dist = float('inf')
        l1 = [min_dist, 0]  # (distance of point, predicted class)
        l2 = l1
        l3 = l1
        foldNo = 0
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
            """iterate through dict to find closest 5 points"""
            for dist in f_dists[f1vec_index]: #for vector 0
                if (dist <= l1[0]):
                    l1[0] = dist
                    l1[1] = classes[foldNo-1][fvec_index]
                elif (dist > l1[0] and dist < l2[0]):
                    l2[0] = dist
                    l2[1] = classes[foldNo-1][fvec_index]
                elif (dist > l2[0] and dist < l3[0]):
                    l3[0] = dist
                    l3[1] = classes[foldNo-1][fvec_index]
                fvec_index = fvec_index + 1
        vArr[f1vec_index] = most_frequent_class([l1,l2,l3])
        f1vec_index = f1vec_index + 1
    return vArr


# Returns Dict with keys "fold'No'_features" : items = tuple ([class of closest point], [class of 2nd closest point]...)
def knn3():
    foldNo = 0
    vClass_dict = {}
    while (foldNo < 10):
        foldNo = foldNo + 1
        foldname = "fold" + str(foldNo)+"_features"
        f = data.get(foldname)
        start = time.time()
        vClass_dict[foldname] = computeFoldClasses3(f,foldNo)
        end = time.time()
        print("completed "+str(foldNo)+" in time:")
        print(end - start)
    return vClass_dict

"""Takes the dictionary returned from knn1() and builds a confusion matrix."""
def knn3ConfMatrix(dict):
    # Rows = class i, 1-10
    # columns = classified as class j
    matrix = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
    for key in dict.keys():
        i = 0
        for class_no in dict[key]: #class_no = dict["fold1_features"][i]
            classskey = key[:-8] + "classes"
            real_class = data[classskey][i][0] #real class = actual class
            predicted_class = class_no
            matrix[real_class-1][predicted_class-1] += 1
            i +=1
    return matrix

"""Takes the dictionary returned from knn and builds a confusion matrix."""
def confMatrix(dict):
    # Rows = class i, 1-10
    # columns = classified as class j
    matrix = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
    for key in dict.keys():
        i = 0
        for class_no in dict[key]: #class_no = dict["fold1_features"][i]
            classskey = key[:-8] + "classes"
            real_class = data[classskey][i][0] #real class = actual class
            predicted_class = class_no
            matrix[real_class-1][predicted_class-1] += 1
            i +=1
    return matrix

"""Additional useful methods:"""
"""Given a list of tuples of form(dictance, class),
find most common class and returns it. If clash, then closest point."""
def most_frequent_class(lsts):
    count_class = [0,0,0,0,0,0,0,0,0,0]
    #Count occurence of classes
    for lst in lsts:
        count_class[(lst[1]-1)] += 1

    class_clashes = []
    max_occ = max(count_class)
    #Create a list of most frequent classes that occur in class_clashes
    for i in range(len(count_class)):
        if (count_class[i] == max_occ):
            # Now we found a clash
            class_clashes.append(i+1)

    min_dist = float('inf')
    most_frequent = class_clashes[0]

    for j in range(len(class_clashes)):
        for lst in lsts:
            if (lst[1] == class_clashes[j]):
                if (lst[0] <= min_dist):
                    min_dist = lst[0]
                    most_frequent = class_clashes[j]
    return most_frequent
