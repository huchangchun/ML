#encoding=utf-8
import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
import math
import numpy as np

from Util.Metas import TimingMeta

class Cluster(metaclass=TimingMeta):
    def __init__(self, x, y, sample_weight=None, base = 2):
        self._x, self._y = x.T, y
        if sample_weight is None:
            self._counters = np.bincount(self._y)
        else:
            self._counters = np.bincount(self._y,weights=sample_weight * len(sample_weight))
        
        self._sample_weight = sample_weight
        self._con_chaos_cache = self._ent_cache = self._gini_cache = None
        self._base = base
    def __str__(self):
        return "Cluster"
    
    __repr__ = __str__
    #定义信息熵
    def ent(self, ent=None, eps = 1e-12):
        if self._ent_cache is  not None and ent is None:
            return self._ent_cache
        y_len = len(self._y)
        if ent is None:
            ent = self._counters
        #h(y) = -sum(p_k * log(p_k))
        ent_cache = max(eps, _sum([c / y_len * math.log(c / y_len, self._base) if c != 0 else 0 for c in ent ]))
        if ent is None:
            self._ent_cache = ent_cache
        return ent_cache
    #定义基尼系数
    def gini(self, p=None):
        if self._gini_cache is not None and p is None:
            return self._gini_cache
        if p is None:
            p = self._counters
        #g(y) =1 -sum(p_k^2)
        gini_cache = 1 - np.sum((p / len(self._y)) ** 2)
        if p is None:
            self._gini_cache = gini_cache
        return gini_cache
    
    #计算条件熵h(y|A)和条件基尼g(y|A)
    def bin_con_chaos(self, idx, tar, criterion="gini", continuous=False):
        """
        idx: 样本特征维度，
        tar: idx维度下
        定义计算条件熵的函数
        获取指定维数特征A的分类mask
        统计每个分类的个数
        获取在特征A限制下的^y
        遍历分类mask,和^y
            用mask划分x,留下该维度为特征ai的数据集sub_ai
            计算sub_ai的信息熵/基尼系数r
            累加 该数据集的概率与r的乘积
        """
        if criterion == "ent":
            method = lambda cluster: cluster.ent()
        elif criterion == "gini":
            method = lambda cluster: cluster.gini()
        else:
            raise NotImplementedError("Conditional info criterion '{}' not defined".format(criterion))
        data = self._x[idx]
        tar = data == tar if not continuous else data < tar
        tmp_labels = [tar, ~tar] 
        self._con_chaos_cache = [np.sum(label) for label in tmp_labels]
        label_lst = [self._y[label] for label in tmp_labels]
        rs, chaos_lst, xt = 0,[],self._x.T
        append = chaos_lst.append
        for label_mask, tar_label in zip(tmp_labels, label_lst):
            tmp_data = xt[label_mask] #xt(12,4) label_mask(12,)
            if self._sample_weight is None:
                chaos = method(Cluster(tmp_data, tar_label, base=self._base))
            else:
                new_weights = self._sample_weight[label_mask]
                chaos = method(Cluster(tmp_data, tar_label, new_weights / np.sum(new_weights), base=self._base))
            rs += len(tmp_data) / len(data) * chaos
        return rs, chaos_lst
    
    #计算信息增益 
    def bin_info_gain(self, idx, tar, criterion="gini", get_chaos_lst=False, continuous=False):
        """
        idx: 样本数据的维度
        tar: idx维度下的特征ai(i~n)
        根据criterion选择条件熵或者条件基尼系数
        信息增益=总的不确定性-条件下的不确定性
        """
        if criterion in ("ent", "ratio"):
            con_chaos, chaos_lst = self.bin_con_chaos(idx, tar, "ent", continuous)
            gain = self.ent() - con_chaos #信息增益 = h(y) - h(y|A)
            if criterion == "ratio":#信息增益/h_A(y)
                gain = gain / self.ent(self._con_chaos_cache) #h_A(y) = -sum(p)log(p) ,(p(y^A=a_j))
        elif criterion == "gini":
            con_chaos, chaos_lst = self.bin_con_chaos(idx, tar, "gini", continuous)
            gain = self.gini() - con_chaos
        else:
            raise NotImplementedError("Info_gain criterion'{}' not defined".format(criterion))
        return (gain, chaos_lst) if get_chaos_lst else gain
    