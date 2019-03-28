#encoding=utf-8
import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
import time

from c_CvDTree.Tree import *
from Util.Util import DataUtil

"""
决策树小结：
1.获取数据x,y
2.定义一个CartTree树实例：实例对象中比较重要的属性是根节点root并指向了一个CartNode类型的节点
同时定义这棵树的最大树深，
3.调用fit进行决策树的训练：

主要逻辑步骤,以cart树分析：
 1).给根节点输入数据，根节点定义了树
 2).根据不同特征维度的不同特征类别计算以该类别做划分后的信息增益,得到增益最大的维度和特征，
    树的每一层记录了上一层划分的特征和当前最大增益的维度和特征
 3).根绝2得到的维度和特征进行数据划分数据为左右两部分(左边为该特征部分，右边为其他特征)，再调用2,直到达到停止条件(4)
 4).停止条件有达到最大树深度停止，数据维度为0停止,信息增益小于设定的阈值，停止划分时进行第5)
 5).根据数据个数类别作为数据的类别
预测的逻辑过程：
 1).获取数据
 2).循环读取每一个样本
 3).遍历树保存的最大增益维度i和特征ai,，如果样本的第i维特征为ai,则继续往左子树探测，直到子节点的类别存在，返回类别
 如果第i维特征不等于ai，则往右子树探测，直到返回类别
 4).
 剪枝的过程：
 略
"""
def main(visualize=True):
    x, y = DataUtil.get_dataset("balloon.0(en)", "../Data/balloon1.0(en).txt") #(12.4)(12,)
    #x, y = DataUtil.get_dataset("test", "../_Data/test.txt")
    fit_time = time.time()
    tree = CartTree(whether_continuous=[False] * 4)
    tree.fit(x, y, train_only=False)
    fit_time = time.time() - fit_time
    if visualize:
        tree.view()
    estimate_time = time.time()
    tree.evaluate(x, y)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    
    if visualize:
        tree.visualize()

    train_num = 6000
    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "mushroom", "../Data/mushroom.txt", tar_idx=0, n_train=train_num)
    fit_time = time.time()
    tree = C45Tree()
    tree.fit(x_train, y_train)
    fit_time = time.time() - fit_time
    if visualize:
        tree.view()
    estimate_time = time.time()
    tree.evaluate(x_train, y_train)
    tree.evaluate(x_test, y_test)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    if visualize:
        tree.visualize()

    x, y = DataUtil.gen_xor(one_hot=False)
    fit_time = time.time()
    tree = CartTree()
    tree.fit(x, y, train_only=True)
    fit_time = time.time() - fit_time
    if visualize:
        tree.view()
    estimate_time = time.time()
    tree.evaluate(x, y, n_cores=1)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    if visualize:
        tree.visualize2d(x, y, dense=1000)
        tree.visualize()

    wc = [False] * 16
    continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for _cl in continuous_lst:
        wc[_cl] = True

    train_num = 2000
    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "bank1.0", "../Data/bank1.0.txt", n_train=train_num, quantize=True)
    fit_time = time.time()
    tree = CartTree()
    tree.fit(x_train, y_train)
    fit_time = time.time() - fit_time
    if visualize:
        tree.view()
    estimate_time = time.time()
    tree.evaluate(x_test, y_test)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    if visualize:
        tree.visualize()
 
    tree.show_timing_log()   
if __name__=="__main__":
    main(True)