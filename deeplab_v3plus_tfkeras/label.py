import pandas as pd
import numpy as np

class Label():
    def __init__(self,
                 label_file_path):
        label_pd = pd.read_csv(label_file_path, header=None)
        self.name = list(label_pd.iloc[:,0])
        self.color = np.array(label_pd.iloc[:,1:])
        self.n_labels = len(self.name)
