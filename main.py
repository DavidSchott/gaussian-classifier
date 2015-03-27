__author__ = 's1329380'
import gaussian,knn

class main:
    def __init__(self):
        #Confusion Matrices
        self.zlist = [0,0,0,0,0,0,0,0,0,0]
        #Accuracies:
        self.knn1_acc = 0.0
        self.knn3_acc = 0.0
        self.knn5_acc = 0.0
        self.knn1_conf = [self.zlist for i in range(10)]
        self.knn3_conf = [self.zlist for i in range(10)]
        self.knn5_conf = [self.zlist for i in range(10)]

        self.gauss_full_acc = 0.0
        self.gauss_shared_acc = 0.0
        self.gauss_full_conf_matrix = [self.zlist for i in range(10)]
        self.gauss_shared_conf_matrix = [self.zlist for i in range(10)]
    """Generates all confusion matrices+accs for each knn-classification and stores them conveniently into main."""
    def knn_full(self):
        knn.getAccAll()
        self.knn1_acc = knn.knn1_acc
        self.knn3_acc = knn.knn3_acc
        self.knn5_acc = knn.knn5_acc
        self.knn1_conf = knn.knn1_conf
        self.knn3_conf = knn.knn3_conf
        self.knn5_conf = knn.knn5_conf
    """Generates all confusion matrices+accs for each Gauss-classification and stores them conveniently into main."""
    def gauss_full(self):
        self.gauss_full_acc = gaussian.gauss_full_acc
        self.gauss_shared_acc = gaussian.gauss_shared_acc
        self.gauss_shared_conf_matrix = gaussian.gauss_shared_conf
        self.gauss_full_conf_matrix = gaussian.gauss_full_conf
