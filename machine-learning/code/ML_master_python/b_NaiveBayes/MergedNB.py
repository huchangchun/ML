#encoding=utf-8
import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
from b_NaiveBayes.Basic import *
from b_NaiveBayes.MultinomialNB import MultinomialNB
from b_NaiveBayes.GaussianNB import GaussianNB

from Util.Util import DataUtil
from Util.Timing import Timing

class MergedNB(NaiveBayes):
    MergedNBTiming = Timing()
    
    def __init__(self, **kwargs):
        super(MergedNB, self).__init__(**kwargs)
        #定义两个实例分别用与处理连续和离散数据
        self._multinomial, self._gaussian = MultinomialNB(), GaussianNB()
        
        wc = kwargs.get("whether_continuous")#通过**kwargs传递字典进来，通过get("key")获取指定的value
        if wc is None:
            self._whether_discrete = self._whether_continuous = None
        else:
            self._whether_continuous = np.asarray(wc)
            self._whether_discrete = ~self._whether_continuous
    @MergedNBTiming.timeit(level=1, prefix="[API] ")
    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        x, y, wc, features, feat_dicts, label_dict = DataUtil.quantize_data(x, y, wc = self._whether_continuous, separate=True)
        if self._whether_continuous is None:
            self._whether_continuous = wc
            self._whether_discrete = ~self._whether_continuous
        self.label_dict = label_dict
        discrete_x, continuous_x = x
        
        cat_counter = np.bincount(y)
        self._cat_counter = cat_counter
        
        labels = [y == value for value in range(len(cat_counter))]
        labelled_x =[discrete_x[ci].T for ci in labels]
        
        #对离散型数据初始化离散型朴素贝叶斯
        self._multinomial._x, self._multinomial._y = x, y
        self._multinomial._labelled_x, self._multinomial._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._multinomial._cat_counter = cat_counter
        self._multinomial._feat_dicts = [dic for i,dic in enumerate(feat_dicts) if self._whether_discrete[i]]
        self._multinomial._n_possibilities = [len(feats) for i, feats in enumerate(features) if self._whether_discrete[i]]
        self._multinomial.label_dict = label_dict
        
        #对连续型数据初始化连续型朴素贝叶斯
        labelled_x = [continuous_x[label].T for label in labels]
        self._gaussian._x, self._gaussian._y = continuous_x.T, y
        self._gaussian._labelled_x, self._gaussian._label_zip = labelled_x, labels
        self._gaussian._cat_counter, self._gaussian.label_dict = cat_counter, label_dict
        self.feed_sample_weight(sample_weight)
        
    @MergedNBTiming.timeit(level=1, prefix="[Core] ")
    def feed_sample_weight(self, sample_weight=None):
        self._multinomial.feed_sample_weight(sample_weight)
        self._gaussian.feed_sample_weight(sample_weight)
    @MergedNBTiming.timeit(level=1, prefix="[Core] ")
    def _fit(self, lb):
        """
        调用离散和连续朴素贝叶斯对象分别进行训练
        取先验概率
        """
        self._multinomial.fit()
        self._gaussian.fit()
        self._p_category = self._multinomial["p_category"]
    
    @MergedNBTiming.timeit(level=1, prefix="[Core] ")
    def _func(self, x, i):
        """
        数据转换为2d，将连续型数据的预测概率乘离散型数据的预测概率得到整体的后验概率，因为两个都乘了先验，所以要除一个先验
        """
        x = np.atleast_2d(x)
        return self._multinomial["func"](
            x[:, self._whether_discrete].astype(np.int), i) * self._gaussian["func"](
            x[:, self._whether_continuous], i) / self._p_category[i]
    @MergedNBTiming.timeit(level=1, prefix="[Core] ")
    def _transfer_x(self, x):
        """
        数据转化
        对于离散型数据，将字符转化为数字
        对于连续型数据，将数据转化为float
        """
        feat_dicts = self._multinomial["feat_dicts"]
        idx = 0
        #因为离散型数据居多，所以遍历离散的
        for dim, discrete in enumerate(self._whether_discrete):
            for i, sample in enumerate(x):
                if not discrete:#如果是连续的，则转化为float()
                    x[i][dim] = float(x[i][dim])
                else:#如果是离散的，则查字典
                    x[i][dim] = feat_dicts[idx][sample[dim]]
            if discrete: #因为feat_dicts只对应离散的数据字典，所以用一次加1
                idx += 1
        return x
    

if __name__ == '__main__':
    import time

    whether_continuous = [False] * 16
    continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for cl in continuous_lst: #查看数据集，设置对应的特征是否连续还是离散
        whether_continuous[cl] = True

    train_num = 40000

    data_time = time.time()
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset(
        "bank1.0", "../Data/bank1.0.txt", n_train=train_num)
    data_time = time.time() - data_time

    learning_time = time.time()
    nb = MergedNB(whether_continuous=whether_continuous)
    nb.fit(x_train, y_train)
    learning_time = time.time() - learning_time

    estimation_time = time.time()
    nb.evaluate(x_train, y_train)
    nb.evaluate(x_test, y_test)
    estimation_time = time.time() - estimation_time
    nb.show_timing_log()