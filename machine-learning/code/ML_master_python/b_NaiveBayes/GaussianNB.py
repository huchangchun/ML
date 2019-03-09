import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
import matplotlib.pyplot as plt
from b_NaiveBayes.Basic import *
from Util.Timing import Timing

class GaussianNB(NaiveBayes):
    GaussianNBTiming = Timing()
    
    @GaussianNBTiming.timeit(level=1, prefix="[API] ")
    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        x = np.array([list(map(lambda c: float(c),sample)) for sample in x])
        labels = list(set(y))
        label_dict = {label: i for i,label in enumerate(labels)}
        y = np.array([label_dict[yy] for yy in y])
        cat_counter = np.bincount(y)
        labels = [y == value for value in range(len(cat_counter))] #获取分类
        labelled_x =[x[label].T for label in labels] #labelled_x对应样本按y分类后的数据集合，子集为各个类别的样本
        
        self._x, self._y = x.T, y
        self._labelled_x, self._label_zip = labelled_x, labels
        self._cat_counter, self.label_dict = cat_counter, {i:l for i,l in label_dict.items()}
        self.feed_sample_weight(sample_weight)
        
    @GaussianNBTiming.timeit(level=1, prefix="[Core] ")
    def feed_sample_weight(self, sample_weight=None):
        if sample_weight is not None:
            local_weight = sample_weight * len(sample_weight)
            for i, label in enumerate(self._label_zip):
                self._labelled_x[i] *= local_weight[label]
    @GaussianNBTiming.timeit(level=1, prefix="[Core] ")
    def _fit(self, lb):
        lb = 0
        n_catagory = len(self._cat_counter)
        self._p_category = self.get_prior_probability(lb)
        data = [
            NBFunctions.gaussion_maximum_likelihood(self._labelled_x, n_category, dim) for dim in rangel(len(self._x))
        ]
        self._data = data
        