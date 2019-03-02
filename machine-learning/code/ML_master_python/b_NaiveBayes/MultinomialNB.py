#encoding=utf-8
import numpy as np
import time
import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
from b_NaiveBayes.Basic import *
from Util.Util import DataUtil
from Util.Timing import Timing

class MultinomialNB(NaiveBayes):
    MultinomialNBTiming = Timing()
    
    @MultinomialNBTiming.timeit(level=1, prefix="[API] ")
    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        x, y, _, features, feat_dicts, label_dict = DataUtil.quantize_data(x, y, wc=np.array([False] * len(x[0])))
        cat_counter=np.bincount(y)
        n_possibilities = [len(feats) for feats in features] #记录各维度特征的取值个数
        labels = [y == value for value in range(len(cat_counter)) ]#获取各类别的数据的下标
        labelled_x = [x[ci].T for ci in labels]
    @MultinomialNBTiming.timeit(level=2, prefix="[API] ")
    def fit(self, x=None, y=None, sample_weight=None, lb=None):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]
        if lb is None:
            lb = self._params["lb"]
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weight)
        self._fit(lb)
   
    
    @MultinomialNBTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_result=False, **kwargs):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        else:
            x = [xx[:] for xx in x]
        x = self._transfer_x(x)
        
        m_arg, m_probability = np.zeros(len(x), dtype=np.int8), np.zeros(len(x))
        for i in range(len(self._cat_counter)):
            p = self._func(x,i)
            mask = p > m_probability
            m_arg[mask], m_probability[mask] = i, p[mask]
        if not get_raw_result:
            return np.array([self.label_dict[arg] for arg in m_arg])
        return m_probability        
        
if __name__=="__main__":
    train_num = 6000
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset("mushroom","../Data/mushroom.txt", n_train=train_num,tar_idx=0)
    learning_time = time.time()
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    learning_time = time.time() - learning_time
    estimation_time = time.time()
    nb.evaluate(x_train, y_train)
    nb.evaluate(x_test, y_test)
    estimation_time = time.time() - estimation_time
    print("Model building  : {:12.6} s\n"
          "Estimation      : {:12.6} s\n"
          "Total           : {:12.6} s\n".format(
              learning_time, estimation_time,learning_time+estimation_time
          )  
    )
    nb.show_timing_log()
    #nb.visualize()