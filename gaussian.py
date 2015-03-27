__author__ = 's1329380'
#import scipy.io, math, pprint
import numpy as np
import knn,time, MyMean,MyCov
import scipy as sp

gauss_shared_conf = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
gauss_full_conf = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]

gauss_shared_acc = 0.0
gauss_full_acc = 0.0

"""Returns an dictionary with keys "classNo", containing a list of all testing vectors belonging to that foldNo-class"""
def classDict(foldNo):
    dict = {}
    for cl_no in range(1, 11):
        list_temp = []
        for j in range(1, 11):
            key = "fold"+str(j)
            vectors = knn.data[key+"_features"]    # this is one fold_features with 5000 vectors
            real_class = knn.data[key+"_classes"]  # this is the corresponding fold_class
            for i in range(len(vectors)):
                if (real_class[i][0] == cl_no and (j != foldNo)):
                    list_temp.append(vectors[i])
        dict[cl_no] = (list_temp)               #a list of lists of vectors, for each clno
    return dict

"""----------------------------BEGINNING OF FULL GAUSSIAN MODEL MATRIX CALCULATIONS----------------------------"""

"""Generates a dict with keys classNo and corresponding values. Values = (Cov, mean, Inv(Cov), logdet(Cov))"""
def fullGaussDict(foldNo):
    dict = classDict(foldNo)
    gaussdict = {} # Keys are class-no's. Here we store important values in form (mean,cov,cov^-1, logdet(cov))
    for key in dict.keys():
        mean_temp = MyMean.mean(dict[key])
        cov_temp = MyCov.cov(dict[key], mean_temp)
        inv_temp = sp.linalg.inv(cov_temp,False,True)
        (sign, logdet) = np.linalg.slogdet(cov_temp)
        gaussdict[key] = (mean_temp,cov_temp,inv_temp,logdet)
    return gaussdict

def log_prob_classGivenVec(vec,tuple):
    #Step 1: -0.5 * (vec - mean)
    diff = (vec - tuple[0])
    step1 = (diff) * -0.5

    #Step 2: cov^-1 *(vec-mean)^T
    step2 = np.dot(tuple[2], np.transpose(diff))

    #Step 3: step1 * step2
    step3 = np.dot(step1,step2)

    #Step 4: -0.5 * logdet(cov)
    step4 = -0.5 * tuple[3]

    #Step 5: step4 + step3 + ln(0.1)
    step5 = np.add(step4, step3) + np.log(0.1)
    return step5

"""Returns an dict with keys vec-number and items (probability classified as class, class)"""
def fullGaussFold(f,foldNo):
    gauss = fullGaussDict(foldNo)
    probs = {}
    #loop through each vector in f
    for i in range(len(f)):
        # (probability, classification)
        tuple = (0.0,0)
        prob_max = 0.0
        prob_temp = 0.0
        #find largest classification.
        for cl_no in range(1, 11):
            prob_temp = log_prob_classGivenVec(f[i], gauss[cl_no])
            if (prob_temp > prob_max):
                prob_max = max(prob_temp, prob_max)
                tuple = (prob_max, cl_no)
        probs[str(i+1)] = tuple
    return probs


""" Returns an dictionary with key "fold'No'_features", corr. item = dict[vecno] = (probability classified as class, predicted class)"""
def fullGaussTotal():
    foldNo = 0
    #Dict with keys "fold'No'_features" : items = array [given vector of fold'No'_features, corresponding classification number]
    vClass_dict = {}
    while (foldNo < 10):
        foldNo = foldNo + 1
        foldname = "fold" + str(foldNo)+"_features"
        f = knn.data.get(foldname)
        start = time.time()
        vClass_dict[foldname] = fullGaussFold(f,foldNo)
        end = time.time()
        print("completed "+str(foldNo)+" in time:")
        print(end - start)
    return vClass_dict


"""----------------------------BEGINNING OF SHARED GAUSSIAN MODEL MATRIX CALCULATIONS----------------------------"""
"""Returns the shared covariance of one fold, by summing up all covariances of other folds and dividing by N"""
def sharedCov(foldNo):
    cov = [[0 for i in range(len(knn.data["fold1_features"][0]))] for j in range(len(knn.data["fold1_features"][0]))]
    dict = classDict(foldNo) # generate classdict of vectors of other folds, classified by classes
    for key in dict.keys():
        mean_temp = MyMean.mean(dict[key])
        cov_temp = MyCov.cov(dict[key], mean_temp)
        cov = np.add(cov, cov_temp)
    return cov * 0.1

"""Generates a dict with keys classNo and corresponding values. Values = (Cov, mean, Inv(Cov), logdet(Cov))"""
def sharedGaussDict(foldNo):
    dict = classDict(foldNo)
    cov = sharedCov(foldNo)
    inv_temp = sp.linalg.inv(cov,False,True)
    (sign, logdet) = np.linalg.slogdet(cov)
    gaussdict = {} # Keys are class-no's. Here we store core values in form (mean,cov,cov^-1, logdet(cov))
    for key in dict.keys():
        mean_temp = MyMean.mean(dict[key])
        gaussdict[key] = (mean_temp,cov,inv_temp,logdet)
    return gaussdict

"""This function is identical to the pClassGivenVec, except that we do not compute Step 4"""
def linearDiscrim(vec,tuple):
    return log_prob_classGivenVec(vec,tuple)

"""Returns an dict with keys vec-number and items (probability classified as class, class)"""
def sharedGauss(f,foldNo):
    gauss = sharedGaussDict(foldNo)
    probs = {}
    #loop through each vector in f
    for i in range(len(f)):
        # (probability, classification)
        tuple = (0.0,0)
        prob_max = 0.0
        prob_temp = 0.0
        #find largest classification.
        for cl_no in range(1, 11):
            prob_temp = linearDiscrim(f[i], gauss[cl_no])
            if (prob_temp > prob_max):
                prob_max = max(prob_temp, prob_max)
                tuple = (prob_max, cl_no)
        probs[str(i+1)] = tuple
    return probs

""" Returns an dictionary with key "fold'No'_features", corr. item = dict[vecno] = (probability classified as class, predicted class)"""
def sharedGaussTotal():
    foldNo = 0
    #Dict with keys "fold'No'_features" : items = array [given vector of fold'No'_features, corresponding classification number]
    vClass_dict = {}
    while (foldNo < 10):
        foldNo = foldNo + 1
        foldname = "fold" + str(foldNo)+"_features"
        f = knn.data.get(foldname)
        start = time.time()
        vClass_dict[foldname] = sharedGauss(f,foldNo)
        end = time.time()
        print("completed "+str(foldNo)+" in time:")
        print(end - start)
    return vClass_dict


"""----------------------------CREATION OF CONFUSION MATRIX----------------------------"""
"""Builds the confusionMatrix using dict recovered from dict with key "foldno_features" and prob. of item in tuple[1]"""
def getConfMatrix(dicts):
    # Rows = class i, 1-10
    # columns = classified as class j
    matrix = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
    for key in dicts.keys():
        for i in (range(0,5000)):
            classskey = key[:-8] + "classes"
            real_class = knn.data[classskey][i][0] -1
            key_i = str(i+1)
            tuple = dicts[key][key_i]
            predicted_class = tuple[1] - 1 #used to be tuple
            matrix[real_class][predicted_class] += 1
    return matrix

#Returns the total classification accuracy for gaussian classification
def confMatrixAcc(conf_matrix):
    sum = 0.0
    for i in range(len(conf_matrix)):
        sum += conf_matrix[i][i]
    return sum / 50000.0


"""All at once"""
def gaussAll():
    global gauss_shared_conf
    global gauss_full_conf
    global gauss_shared_acc
    global gauss_full_acc
    gauss_shared_conf = getConfMatrix(sharedGaussTotal())
    gauss_full_conf = getConfMatrix(fullGaussTotal())
    gauss_shared_acc = confMatrixAcc(gauss_shared_conf)
    gauss_full_acc = confMatrixAcc(gauss_full_conf)
