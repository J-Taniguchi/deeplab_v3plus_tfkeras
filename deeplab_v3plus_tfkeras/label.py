import pandas as pd
import numpy as np

class Label():
    def __init__(self,
                 label_file_path):
        label_pd = pd.read_csv(label_file_path, header=None)
        self.name = list(label_pd.iloc[:,0])
        self.color = np.array(label_pd.iloc[:,1:])
        self.n_labels = len(self.name)
        self.class_weight = [1/self.n_labels] * self.n_labels
        print("labels", self.name)

    def add_class_weight(self, class_weight):
        if len(class_weight) != self.n_labels:
            raise Exception("list of class_weight length must be equal to n_labels")
        else:
            sum_weights = sum(class_weight)
            if sum_weights != 1.0:
                print("sum of class_weight is not 1.0. Therefore, set as such.")
                print("input class_weight list: ", class_weight)
                for i in range(len(class_weight)):
                    class_weight[i] /= sum_weights
                print("modified class_weight list: ", class_weight)
            else:
                print("input class_weight list: ", class_weight)
        self.class_weight = class_weight
