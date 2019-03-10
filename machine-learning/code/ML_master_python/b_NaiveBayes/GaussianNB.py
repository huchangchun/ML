import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
import matplotlib.pyplot as plt
from b_NaiveBayes.Basic import *


from Util.Timing import Timing
from Util.Util import DataUtil
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
        n_category = len(self._cat_counter)
        self._p_category = self.get_prior_probability(lb)
        data = [
             
            NBFunctions.gaussian_maximum_likelihood(self._labelled_x, n_category, dim) for dim in range(len(self._x))
        ]
        self._data = data
    @GaussianNBTiming.timeit(level=1, prefix="[Core] ")
    def _func(self, x, i):
        x = np.atleast_2d(x).T
        rs = np.ones(x.shape[1])
        for d, xx in enumerate(x):
            rs *= self._data[d][i](xx)
        return rs * self._p_category[i]
    def visualize(self, save=False):
        colors = plt.cm.Paired([i / len(self.label_dict) for i in range(len(self.label_dict))])
        colors = {cat: color for cat, color in zip(self.label_dict.values(), colors)}
        for j in range(len(self._x)):
            tmp_data = self._x[j]
            x_min, x_max = np.min(tmp_data),np.max(tmp_data)
            gap = x_max - x_min
            tmp_x = np.linspace(x_min - 0.1 * gap, x_max + 0.1 * gap, 200)
            title = "$j = {}$".format(j + 1)
            plt.figure(num=None, figsize=None, dpi=None, facecolor=None, 
                      edgecolor=None, frameon=True, 
                      FigureClass=Figure)
            plt.title(title)
            for c in range(len(self.label_dict)):
                plt.plot(tmp_x, self._data[j][c](tmp_x),
                         c = colors[self.label_dict[c]], label="class: {}".format(self.label_dict[c]))
                plt.xlim(x_min - 0.2 * gap, x_max + 0.2 * gap)
                plt.legend()
                if not save:
                    plt.show()
                else:
                    plt.savefig("d{}".format(j + 1))
if __name__=="__main__":
    import time
   
    train_num = 6000
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset(
        "mushroom", "../Data/mushroom.txt", n_train=train_num, tar_idx=0)

    learning_time = time.time()
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    learning_time = time.time() - learning_time
    estimation_time = time.time()
    nb.evaluate(x_train, y_train)
    nb.evaluate(x_test, y_test)
    estimation_time = time.time() - estimation_time
    print("Model build  :  {:12.6} s\n"
          "Estimation   :  {:12.6} s\n"
          "Total        :  {:12.6} s\n".format(learning_time, estimation_time,
                                               learning_time + estimation_time) 
    )
   
    nb.show_timing_log()
    nb.visualize()