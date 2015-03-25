__author__ = 's1329380'
#import scipy.io, math, pprint
import numpy as np
import knn
import scipy as sp

#Sum of columns / no of rows. Only works for np-Matrix.
def mean(m1):
    sum_cols = m1.sum(axis = 0)
    mean_cols = sum_cols
    for sum in sum_cols:
        mean_cols = sum / len(m1)
    return mean_cols

def MyMean(vec_feat):
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
def gaussianModel(foldNo):
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
    #np.subtract(vec, tuple[0])
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
    gauss = gaussianModel(foldNo)
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
            print("Completed class "+str(cl_no)+" of vector "+str(i+1)) #used for testing
        probs[str(i+1)] = tuple
        print("VECTOR "+str(i+1) + " COMPLETED")
    return probs

def gaussAcc(f,foldNo):
    tupledict = foldloglikehood(f,foldNo)
    j = 1
    for i in range(len(f)):
        real_class = knn.data["fold"+str(foldNo)+"_classes"][i]
        if (tupledict[str(i+1)][1] == real_class[0]):
            j += 1
    return j/5000.0


f1 = knn.data.get("fold1_features")
v1 = f1[0]
#Used for testing
def main():
    m1 = np.asmatrix([[4.,2.,0.6],
                     [4.2,2.1,0.59],
                     [3.9,2.,0.58],
                     [4.3,2.1,0.62],
                     [4.1,2.2,0.63]])
    return foldloglikehood(f1,1)__author__ = 's1329380'
#import scipy.io, math, pprint
import numpy as np
import knn
import scipy as sp

#Sum of columns / no of rows. Only works for np-Matrix.
def mean(m1):
    sum_cols = m1.sum(axis = 0)
    mean_cols = sum_cols
    for sum in sum_cols:
        mean_cols = sum / len(m1)
    return mean_cols

def MyMean(vec_feat):
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
def gaussianModel(foldNo):
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
    #np.subtract(vec, tuple[0])
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
    gauss = gaussianModel(foldNo)
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
            print("Completed class "+str(cl_no)+" of vector "+str(i+1)) #used for testing
        probs[str(i+1)] = tuple
        print("VECTOR "+str(i+1) + " COMPLETED")
    return probs

def gaussAcc(f,foldNo):
    tupledict = foldloglikehood(f,foldNo)
    j = 1
    for i in range(len(f)):
        real_class = knn.data["fold"+str(foldNo)+"_classes"][i]
        if (tupledict[str(i+1)][1] == real_class[0]):
            j += 1
    return j/5000.0


f1 = knn.data.get("fold1_features")
v1 = f1[0]
#Used for testing
def main():
    m1 = np.asmatrix([[4.,2.,0.6],
                     [4.2,2.1,0.59],
                     [3.9,2.,0.58],
                     [4.3,2.1,0.62],
                     [4.1,2.2,0.63]])
    return foldloglikehood(f1,1)
