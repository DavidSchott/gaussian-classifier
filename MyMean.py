__author__ = 's1329380'
#import scipy.io, math, pprint
import numpy as np
import knn,time
import scipy as sp

#Sum of columns / no of rows. Only works for np-Matrix.
def mean(m1):
    sum_cols = m1.sum(axis = 0)
    mean_cols = sum_cols
    for sum in sum_cols:
        mean_cols = sum / len(m1)
    return mean_cols
"""Takes a set of feature-vectors and returns meanvector"""
def MyMean(feats):
    columnSum = 0
    vec_mean = []
    for i in range(len(feats[0])):
        for j in range(len(feats)):
            columnSum = columnSum + feats[j][i]
        vec_mean.append(columnSum)
        columnSum = 0
    for element in range(len(vec_mean)):
        vec_mean[element] = vec_mean[element] / (len(feats) + 0.0)
    return np.array(vec_mean)

"""Returns the covariance matrix of a set of feature-vectors and corresponding meanvector"""
def MyCov(feats, meanvec):
    cov = [[0 for i in range(len(feats[0]))] for j in range(len(feats[0]))]
    for i in range(len(feats)):
        diff = np.asmatrix(feats[i]) - np.asmatrix(meanvec)
        diffT = diff.transpose()
        cov = cov + (diffT * diff)
    return cov * (1.0/(len(feats)-1.0))


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


"""Generates a dict with keys classNo and corresponding values. Values = (Cov, mean, Inv(Cov), logdet(Cov))"""
def fullGaussianModel(foldNo):
    dict = classDict(foldNo)
    gaussdict = {} # Keys are class-no's. Here we store important values in form (mean,cov,cov^-1, logdet(cov))
    for key in dict.keys():
        mean_temp = MyMean(dict[key])
        cov_temp = MyCov(dict[key], mean_temp)
        inv_temp = sp.linalg.inv(cov_temp,False,True)
        (sign, logdet) = np.linalg.slogdet(cov_temp)
        gaussdict[key] = (mean_temp,cov_temp,inv_temp,logdet)
        print("Completed computing gaussianModel of class" + str(key))
    return gaussdict

"""gaussdict[key+"_mean"] = mean_temp
        gaussdict[key+"_cov"] = cov_temp
        gaussdict[key+"_inv"] = inv_temp
        gaussdict[key+"_det"] = logdet"""

"""Do this 10x for each class. Then call this method 5000 times for each fold.
   returns ln(P(Class = clno | x = vec of fold_foldNo)).
   Parameters: clno = classnumber
               vec = vector
               tuple = (mean,cov, inv(cov),logdet(cov)) of classnumber"""
def pClassgivenVec(vec,tuple):
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
def foldloglikehood(f,foldNo):
    gauss = fullGaussianModel(foldNo)
    probs = {}
    #loop through each vector in f
    for i in range(len(f)):
        # (probability, classification)
        tuple = (0.0,0)
        prob_max = 0.0
        prob_temp = 0.0
        #find largest classification.
        for cl_no in range(1, 11):
            prob_temp = pClassgivenVec(f[i], gauss[cl_no])
            if (prob_temp > prob_max):
                prob_max = max(prob_temp, prob_max)
                tuple = (prob_max, cl_no)
        probs[str(i+1)] = tuple
    return probs

"""Sums up the accuracies of foldloglikehood and returns"""
def covGaussAcc(f,foldNo):
    tupledict = foldloglikehood(f,foldNo)
    j = 1
    for i in range(len(f)):
        real_class = knn.data["fold"+str(foldNo)+"_classes"][i]
        if (tupledict[str(i+1)][1] == real_class[0]):
            j += 1
    return j/5000.0

""" Returns an dictionary with key "fold'No'_features", corr. item = dict[vecno] = (probability classified as class, predicted class)"""
def loglikehoodTotal():
    foldNo = 0
    #Dict with keys "fold'No'_features" : items = array [given vector of fold'No'_features, corresponding classification number]
    vClass_dict = {}
    while (foldNo < 10):
        foldNo = foldNo + 1
        foldname = "fold" + str(foldNo)+"_features"
        f = knn.data.get(foldname)
        start = time.time()
        vClass_dict[foldname] = foldloglikehood(f,foldNo)
        end = time.time()
        print("completed "+str(foldNo)+" in time:")
        print(end - start)
    return vClass_dict

"""Builds the confusionMatrix using dict recovered from loglikehoodTotal"""
def confusionMatrix(dicts):
    # Rows = class i, 1-10
    # columns = classified as class j
    matrix = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
    for key in dicts.keys():
        #i = 0
        for i in (range(0,10)):
        #for d in dicts[key]: #d = d[vecno] = (probability, pred. class)
            classskey = key[:-8] + "classes"
            real_class = knn.data[classskey][i][0] -1
            key_i = str(i+1)
            tuple = dicts[key][key_i]
            predicted_class = tuple[1] - 1 #used to be tuple
            matrix[real_class][predicted_class] += 1
            #i +=1
    return matrix

#Returns the total classification accuracy for knn-classification
def fullCovGaussAcc(conf_matrix):
    sum = 0.0
    for i in range(len(conf_matrix)):
        sum += conf_matrix[i][i]
    return sum / 50000.0

"""Returns the shared covariance of one fold, by summing up all covariances of other folds and dividing by N"""
def sharedCovFold(foldNo):
    cov = [[0 for i in range(len(knn.data["fold1_features"][0]))] for j in range(len(knn.data["fold1_features"][0]))]
    dict = classDict(foldNo) # generate classdict of vectors of other folds, classified by classes
    for key in dict.keys():
        mean_temp = MyMean(dict[key])
        cov_temp = MyCov(dict[key], mean_temp)
        np.add(cov, cov_temp)
    return cov * 0.1


"""Generates a dict with keys classNo and corresponding values. Values = (Cov, mean, Inv(Cov), logdet(Cov))"""
def sharedGaussianModel(foldNo):
    dict = classDict(foldNo)
    cov = sharedCovFold(foldNo)
    gaussdict = {} # Keys are class-no's. Here we store core values in form (mean,cov,cov^-1, logdet(cov))
    for key in dict.keys():
        mean_temp = MyMean(dict[key])
    inv_temp = sp.linalg.inv(cov,False,True)
    (sign, logdet) = np.linalg.slogdet(cov)
    gaussdict[key] = (mean_temp,cov,inv_temp,logdet)
    print("Completed computing shared gaussianModel of class" + str(key))
    return gaussdict

def linearDiscrim(vec,tuple):
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

#CONTINUE HERE
"""Computes the shared covariance matrix for all matrices"""
def sharedCov():
    foldNo = 0
    while (foldNo < 10):
        foldNo = foldNo + 1
        foldname = "fold" + str(foldNo)+"_features"
        f = knn.data.get(foldname)
        cov = sharedCovFold(foldNo)




        end = time.time()
        print("completed "+str(foldNo)+" in time:")
        print(end - start)
    return vClass_dict

#Used for testing
f1 = knn.data.get("fold1_features")
v1 = f1[0]
def main():
    m1 = np.asmatrix([[4.,2.,0.6],
                     [4.2,2.1,0.59],
                     [3.9,2.,0.58],
                     [4.3,2.1,0.62],
                     [4.1,2.2,0.63]])
    return foldloglikehood(f1,1)
