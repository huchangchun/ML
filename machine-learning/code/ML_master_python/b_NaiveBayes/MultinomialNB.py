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
        cat_counter=np.bincount(y) #统计两个类别的个数
        n_possibilities = [len(feats) for feats in features] #记录各维度特征的取值个数
        labels = [y == value for value in range(len(cat_counter)) ]#获取各类别的数据的下标
        labelled_x = [x[ci].T for ci in labels]
        self._x, self._y = x, y
        self._labelled_x, self._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._cat_counter, self._feat_dicts, self._n_possibilities = cat_counter, feat_dicts, n_possibilities
        self.label_dict = label_dict
        self.feed_sample_weight(sample_weight)
        
    @MultinomialNBTiming.timeit(level=1, prefix="[Core] ")
    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = []
        for dim, p in enumerate(self._n_possibilities):
            if sample_weight is None:#统计各个维度下特征的各个类别数
                self._con_counter.append([np.bincount(xx[dim], minlength=p) for xx in self._labelled_x])
            else:
                self._con_counter.append([np.bincount(xx[dim], weights=sample_weight[label] / sample_weight[label].mean(), minlenght=p) 
                                        for label, xx in self._label_zip])
            
        
    @MultinomialNBTiming.timeit(level=1, prefix="[API] ")
    def _fit(self, lb):
        n_dim = len(self._n_possibilities)
        n_category = len(self._cat_counter)
        self._p_category = self.get_prior_probability(lb) #计算先验概率
                                
        data = [[] for _ in range(n_dim)] #初始化特征维度
        for dim, n_possibilities in enumerate(self._n_possibilities):
            data[dim] = [[ ( self._con_counter[dim][c][p] + lb) / (self._cat_counter[c] + lb * n_possibilities) for p in range(n_possibilities)
                          ] for c in range(n_category)]
        self._data = [np.asarray(dim_info) for dim_info in data]
        
    @MultinomialNBTiming.timeit(level=1, prefix="[Core] ")
    def _transfer_x(self, x):
        for i, sample in enumerate(x):
            for j, char in enumerate(sample):
                x[i][j] = self._feat_dicts[j][char]
        return x
    
    @MultinomialNBTiming.timeit(level=1, prefix="[Core] ")
    def _func(self, x, i):
        x = np.atleast_2d(x).T
        rs = np.ones(x.shape[1]) #初始一个概率数组存放每个样本最终的条件概率,#累乘条件概率
        for d, xx in enumerate(x):  
            rs *= self._data[d][i][xx]
        return rs * self._p_category[i] #条件概率乘上先验概率得到后验概率
    
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