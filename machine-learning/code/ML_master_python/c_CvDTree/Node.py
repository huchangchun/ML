#encoding=utf-8
import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
from math import log2

from c_CvDTree.Cluster import Cluster

from Util.Metas import TimingMeta
class CvDNode(metaclass=TimingMeta):
    def __init__(self, tree=None, base=2, chaos=None,depth=0, parent=None, is_root=True, prev_feat="Root",**kwargs):
        """
        self._x = self._y         记录数据集的变量
        self.base, self.chaos     记录对数的低和当前的不确定性
        self.criterion = sefl.category   记录该Node计算信息增益的方法和所属的类别
        self.left_child = self.right_child   针对连续型特征和CART，记录该Node的左右子节点
        self._children, self.leafs  记录该Node的所有子节点和所有下属的叶节点
        self.sample_weight   记录样本权重
        self.wc  记录各个维度的特征是否是连续的列表
        self.parent, self.is_root  记录该Node的父节点以及该Node是否为根节点
        self.feature_dim  记录作为划分标准的特征所对应的维度j
        self.tar  针对连续型特征和CART,记录二分标准
        self.feats  记录该Node能进行选择的作为划分标准的特征的维度
        self._depth, self.prev_feat  记录Node的深度和其父节点的划分标准
        self.is_cart 记录该Node是否使用了CART算法
        self.is_continuous 记录该Node选择的划分标准对应的特征是否连续
        self.affected 
        self.pruned 记录该Node是否已被剪掉，后面实现局部剪枝算法时会用到
        """
        self._x = self._y = None
        self.base, self.chaos = base, chaos
        self.criterion = sefl.category = None
        self.left_child = self.right_child = None
        self._children, self.leafs = {}, {}
        self.sample_weight = None
        self.wc = None
        
        #如果传入了Tree的话，就进行相应的初始化
        self.tree = tree
        if tree is not None:
            #由于数据预处理是由Tree完成的
            #所以各个维度的特征是否是连续型变量也是由Tree记录的
            self.wc = tree.whether_continuous
            tree.nodes.append(self) 
        self.feature_dim,self.tar, self.feats = None, None,[]
        self.parent, self.is_root = parent, is_root
        self._depth, self.prev_feat = depth, prev_feat
        self.is_cart = self.is_continuous = self.affected = self.pruned = False
        
    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)
    def __lt__(self, other):
        return self. prev_feat < other.prev_feat
    def __str__(self):
        return self.__class__.__name__
    __repr__ = __str__
    
    @property
    def info(self):
        if self.category is None:
            return "CvDNode ({}) ({} -> {})".format(self._depth, self.prev_feat, self.feature_dim)
        return "CvDNode ({}) ({} -> {})".format(self._depth, self.prev_feat, self.tree.y_transformer[self.category])
    
    #主要是区分开连续+CART的情况和其余情况
    #有了该属性后，想要获得所有子节点时就不用分情况讨论了
    @property
    def children(self):
        return {"left": self.left_child, "right": self.right_child} if (self.is_cart or self.is_continuous) else self._children
    
    #递归定义height属性
    #叶节点高度都定义为1，其余节点的高度定义为最高的子节点的高度+1
    @property
    def height(self):
        if self.category is not None:
            return 1
        return 1 + max([_child.height if _child is not None else 0 for _child in self.children.values()])
    #定义info_dic属性，它记录了该Node的主要信息
    @property
    def info_dic(self):
        return {"chaos": self.chaos, "y": self._y}
    
    def fit(self, x, y, sample_weight, feature_bound=None, eps=1e-8):
        """
        1.根据划分标准将数据划分为若干份
        2.依次用这若干份数据实例化新Node(新Node即是当前Node的子节点),同时将当前Node的相关信息传递新的Node
        这里需要注意的是，如果划分标准是离散型的特征的话：
         .若算法是ID3或C4.5,需将该特征对应的维度从新的Node的self.feats属性中除去
         .若算法是CART，需要将二分标准从新的Node的二分标准取值集合中除去。
        最后对新的Node调用fit方法，完成递归
        """
        self._x, self._y = np.atleast_2d(x), np.asarray(y)
        self.sample_weight = sample_weight
        if self.stop1(eps):
            return
        cluster = Cluster(self._x, self._y, sample_weight, self.base)
        if self.is_root:
            if self.criterion =="gini":
                self.chaos = cluster.gini()
            else:
                self.chaos = cluster.ent()
        max_gain, chaos_lst = 0, []
        max_feature = max_tar = None
        feat_len = len(self.feats)
        
        
        