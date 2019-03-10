#encoding=utf8
import os
import sys
root_path =os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
    
    
import numpy as np
from math import pi
from Util.Timing import Timing
from Util.Bases import ClassifierBase

sqrt_pi = (2 * pi) ** 0.5

class NBFunctions:
    @staticmethod
    def gaussian(x, mu, sigma):
        """
        定义正太分布的密度函数
        """
        return np.exp(-(x-mu) ** 2 /(2 * sigma ** 2)) /(sqrt_pi * sigma)
    @staticmethod
    def gaussian_maximum_likelihood(labelled_x, n_category, dim):
        mu = [np.sum(
            labelled_x[c][dim]) / len(labelled_x[c][dim]) for c in range(n_category)]
        sigma = [np.sum(
            (labelled_x[c][dim] - mu[c]) ** 2) / len(labelled_x[c][dim]) for c in range(n_category)]

        def func(_c):
            def sub(x):
                return NBFunctions.gaussian(x, mu[_c], sigma[_c])
            return sub

        return [func(_c=c) for c in range(n_category)]
       


class NaiveBayes(ClassifierBase):
    NavieBayesTiming = Timing()
    def __init__(self, **kwargs):
        super(NaiveBayes, self).__init__(**kwargs)
        self._x  = self._y = self._data = None   
        self._n_possibilities = self._p_category = None 
        self._labelled_x = self._label_zip = None
        self._cat_counter = self._con_counter = None
        self.label_dict = self._feat_dicts = None
        self._params["lb"] = kwargs.get("lb",1)
        
    def feed_data(self, x, y, sample_weight=None):
        pass
    def feed_sample_weight(self, sample_weight=None):
        pass
    @NavieBayesTiming.timeit(level=2, prefix="[API] ")
    def get_prior_probability(self, lb=1):
        return [(c_num + lb) / (len(self._y) + lb * len(self._cat_counter)) for c_num in self._cat_counter]
    
    @NavieBayesTiming.timeit(level=2, prefix="[API] ")
    def fit(self, x=None, y=None, sample_weight=None, lb=None):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]
        if lb is None:
            lb = self._params["lb"]
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weight)
        self._fit(lb)
    def _fit(self, lb):
        pass
    def _func(self, x, i):
        pass
    
    @NavieBayesTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_result=False, **kwargs):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        else:
            x = [xx[:] for xx in x]
        x = self._transfer_x(x) #m_arg记录样本分类情况，m_probability记录样本分类后该类别的概率
        m_arg, m_probability = np.zeros(len(x), dtype=np.int8), np.zeros(len(x))
        for i in range(len(self._cat_counter)):#通过对比样本在每个类别的概率，得到样本的分类情况
            p = self._func(x,i)#获取所有样本对应第i类的概率
            mask = p > m_probability #如果p>初始概率,mask对应的位为True
            m_arg[mask], m_probability[mask] = i, p[mask]#m_arg对应True的类别为i,m_probability记录了样本的最大概率
        if not get_raw_result:
            return np.array([self.label_dict[arg] for arg in m_arg])
        return m_probability
    def _transfer_x(self, x):
        return x
    