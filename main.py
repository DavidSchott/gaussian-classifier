__author__ = 's1329380'
import gaussian,knn


"""Alternatively, for knn:
    1. To get confusion matrices:
       pass knn.knn1() or knn.knn3() or knn.knn5() as parameter to
       knn.knn1ConfMatrix(dict), knn.knn3ConfMatrix(dict), knn.knn5ConfMatrix(dict)

    2. To get prediction accuracy:
       pass respective confusion matrix as parameter to knn.predAcc(matrix)


    For Gaussian:
    1. To get confusion matrices:
       pass gaussian.sharedGaussTotal() or gaussian.fullGaussTotal() as parameter to
       gaussian.getConfMatrix(dicts)

    2. To get prediction accuracy:
        pass respective confusion matrix as parameter to
        gaussian.confMatrixAcc(conf_matrix)"""




class main:
    def __init__(self):
        #Confusion Matrices
        self.zlist = [0,0,0,0,0,0,0,0,0,0]
        #knn-data:
        self.knn1_acc = knn.knn1_acc
        self.knn3_acc = knn.knn3_acc
        self.knn5_acc = knn.knn5_acc
        self.knn1_conf = knn.knn1_conf
        self.knn3_conf = knn.knn3_conf
        self.knn5_conf = knn.knn5_conf
        # Gauss-data
        self.gauss_full_acc = 0.0
        self.gauss_shared_acc = 0.0
        self.gauss_full_conf_matrix = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
        self.gauss_shared_conf_matrix = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
    """Generates all confusion matrices+accs for each knn-classification and stores them conveniently into main."""
    def knn_full(self):
        knn.knnAll()
        knn.confMatrixAll()
        knn.getAccAll()
        self.knn1_acc = knn.knn1_acc
        self.knn3_acc = knn.knn3_acc
        self.knn5_acc = knn.knn5_acc
        self.knn1_conf = knn.knn1_conf
        self.knn3_conf = knn.knn3_conf
        self.knn5_conf = knn.knn5_conf
    """Generates all confusion matrices+accs for each Gauss-classification and stores them conveniently into main."""
    def gauss_full(self):
        gaussian.gaussAll()
        self.gauss_full_acc = gaussian.gauss_full_acc
        self.gauss_shared_acc = gaussian.gauss_shared_acc
        self.gauss_shared_conf_matrix = gaussian.gauss_shared_conf
        self.gauss_full_conf_matrix = gaussian.gauss_full_conf

