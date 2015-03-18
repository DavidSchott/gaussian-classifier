__author__ = 's1329380'
__author__ = 's1329380'
import scipy.io, math, pprint
import numpy as np

#Sum of columns / no of rows
def myMean(m1):
    sum_cols = m1.sum(axis = 0)
    mean_cols = sum_cols
    for sum in sum_cols:
        mean_cols = sum / len(m1)

    return mean_cols


#Used for testing
def main():
    m1 = np.asmatrix([[4.,2.,0.6],
                     [4.2,2.1,0.59],
                     [3.9,2.,0.58],
                     [4.3,2.1,0.62],
                     [4.1,2.2,0.63]])
    print(myMean(m1))