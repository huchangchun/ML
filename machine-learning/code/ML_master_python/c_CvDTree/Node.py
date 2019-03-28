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
        self.criterion = self.category = None
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
            return "CvDNode (depth:{}) (prev:'{}' -> maxdim:{})".format(
                self._depth, self.prev_feat, self.feature_dim)
        return "CvDNode (depth:{}) (prev:'{}' -> class:{})".format(
            self._depth, self.prev_feat, self.tree.y_transformer[self.category])
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
    def info_dict(self):
        return {"chaos": self.chaos, "y": self._y}
    def stop1(self, eps):
        """
        当特征维度为0或者当前Node的数据不确定性小于阈值eps时停止
        同时如果用户指定了决策树的最大树深度，那么当该Node的深度太深时也停止
        若满足了停止条件，该函数会返回True,否则会返回False
        """
        if self._x.shape[1] == 0 or (self.chaos is not None and self.chaos <= eps) or (self.tree.max_depth is not None and self._depth >= self.tree.max_depth):
            self._handle_terminate()
            return True
        return False
    def stop2(self, max_gain, eps):
        if max_gain <= eps:
            self._handle_terminate()
            return True
        return False
    def get_category(self):
        return np.argmax(np.bincount(self._y))
    def _handle_terminate(self):
        """
        首先要生成该Node的所属的类别
        """
        self.category = self.get_category()
        parent = self.parent #然后回溯，更新父节点，父节点的父节点，等等，记录叶节点的属性
        while parent is not None:
            parent.leafs[id(self)] = self.info_dict
            parent = parent.parent
    def prune(self):
        """
        局部剪枝：
        定义一个方法使其能将一个由子节点的Node转化为叶节点
        定义一个方法是使其能挑选出最好的划分标准
        定义一个方法使其能根据划分标准进行生成
        """
        self.category = self.get_category()
        pop_lst = [key for key in self.leafs]
        parent = self.parent
        while parent is not None:
            parent.affected = True
            pop = parent.leafs.pop
            for k in pop_lst:
                pop(k)
            parent.leafs[id(self)] = self.info_dict
            parent = parent.parent
        self.mark_pruned()
        self.feature_dim = None
        self.left_child = self.right_child = None
        self._children = {}
        self.leafs = {}
        
            
    def mark_pruned(self):
        self.pruned = True
        for child in self.children.values():
            if child is not None:
                child.mark_pruned()
    def update_layers(self):
        """
        根据Node的深度，在self.layers对应位置的列表中记录自己
        """
        self.tree.layers[self._depth].append(self)
        for node in sorted(self.children):
            node = self.children[node]
            if node is not None:
                node.update_layers()
    def cost(self, pruned=False):
        if not pruned:
            return sum([leaf["chaos"] * len(leaf["y"]) for leaf in self.leafs.values()])
        return self.chaos * len(self._y)
    def get_threshold(self):
        return (self.cost(pruned=True) - self.cost()) / (len(self.leafs) - 1)
    def cut_tree(self):
        self.tree = None
        for child in self.children.values():
            if child is not None:
                child.cut_tree()
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
        if feature_bound is None:
            indices = range(0, feat_len)
        elif feature_bound == "log":
            indices = np.random.permutation(feat_len)[:max(1, int(log2(feat_len)))]
        else:
            indices = np.random.permutation(feat_len)[:feature_bound]
        tmp_feats = [self.feats[i] for i in indices]
        xt, feat_sets = self._x.T, self.tree.feature_sets
        bin_ig, ig = cluster.bin_info_gain, cluster.info_gain
        for feat in tmp_feats:#[0.1,2,3]遍历每个维度,通过遍历这些维度特征，选取使得不确定性最小的特征
            if self.wc[feat]: #是否连续 wc=whether_continuous
                samples = np.sort(xt[feat])
                feat_set = (samples[:-1] + samples[1:]) *5
            else:#非连续
                if self.is_cart:
                    feat_set = feat_sets[feat] #取第feat维的特征类别 #{'y', 'p'}
                else:
                    feat_set = None
            if self.is_cart or self.wc[feat]:
                for tar in feat_set:
                    tmp_gain, tmp_chaos_lst = bin_ig(feat,tar,criterion = self.criterion, get_chaos_lst=True,continuous=self.wc[feat])
                    if tmp_gain >max_gain:
                        (max_gain, chaos_lst), max_feature,max_tar = (tmp_gain, tmp_chaos_lst), feat, tar
            else:
                tmp_gain, tmp_chaos_lst = ig(feat, self.criterion, True, self.tree.feature_sets[feat])
                if tmp_gain > max_gain:
                    (max_gain, chaos_lst), max_feature = (tmp_gain, tmp_chaos_lst), feat
        if self.stop2(max_gain, eps):
            return
        self.feature_dim = max_feature #max_feature记录划分后获取最大信息增益的维度
        if self.is_cart or self.wc[max_feature]:
            self.tar = max_tar
            #调用根据划分标准进行生成的方法
            self._gen_children(chaos_lst, feature_bound)
            #如果该Node的左子节点和右子节点都是叶节点且所属类别一样，那么就将他们合并，亦进行局部剪枝
            if (self.left_child.category is not None and self.left_child.category == self.right_child.category):
                self.prune()
                #调用Tree的相关方法，将被剪掉的该Node的左右子节点从tree的记录所有Node的列表nodes中除去
                self.tree.reduce_nodes()
        else:
            #调用根据划分标准进行生成的方法
            self._gen_children(chaos_lst,feature_bound)
    def _gen_children(self, chaos_lst, feature_bound):
        feat, tar = self.feature_dim, self.tar #1， ‘l'
        self.is_continuous = continuous = self.wc[feat]
        features = self._x[..., feat] #==self_x[:,feat] #array(['l', 'l', 's', 'l', 's', 'l', 's', 's', 's', 'l', 's', 'l'],
        new_feats = self.feats.copy()
        if continuous:
            mask = features < tar
            masks = [mask, ~mask]
        else:
            if self.is_cart:
                mask = features == tar
                masks = [mask, ~mask] 
                #从特征集合中去除该tar特征
                self.tree.feature_sets[feat].discard(tar)#[{'y', 'p'}, {'l', 's'}, {'a', 'c'}, {'f', 'h'}]
            else:
                masks = None
        if self.is_cart or continuous:#['l','+']
            feats = [tar, "+"] if not continuous else ["{:6.4}-".format(tar), "{:6.4}+".format(tar)]
            for feat, side, chaos in zip(feats,
                                         ["left_child","right_child"],chaos_lst):
                new_node = self.__class__(tree=self.tree, base=self.base, chaos=chaos, 
                                         depth=self._depth + 1, 
                                         parent=self, 
                                         is_root=False, 
                                         prev_feat=feat)
                new_node.criterion = self.criterion
                setattr(self, side, new_node) #直接赋值new_node给self下面的左右孩子
            for node, feat_mask in zip([self.left_child, self.right_child], masks):
                if self.sample_weight is None:
                    local_weights = None
                else:
                    local_weights = self.sample_weight[feat_mask]
                    local_weights /= np.sum(local_weights)
                tmp_data, tmp_labels = self._x[feat_mask,...], self._y[feat_mask]
                if len(tmp_labels) == 0:
                    continue
                node.feats = new_feats
                node.fit(tmp_data,tmp_labels, local_weights, feature_bound)
        else:
            new_feats.remove(self.feature_dim)
            for feat, chaos in zip(self.tree.feature_sets[self.feature_dim], chaos_lst):
                feat_mask = features == feat
                tmp_x = self._x[feat_mask, ...]
                if len(tmp_x) == 0:
                    continue
                new_node = self.__class__(tree=self.tree, base=self.base, chaos=chaos, 
                                         depth=self._depth + 1, 
                                         parent=self, 
                                         is_root=False, 
                                         prev_feat=feat)
                new_node.feats = new_feats
                self.children[feat] = new_node
                if self.sample_weight is None:
                    local_weights = None
                else:
                    local_weights = self.sample_weight[feat_mask]
                    local_weights /= np.sum(local_weights)
                new_node.fit(tmp_x, self._y[feat_mask], local_weights, feature_bound)
                
    def feed_tree(self, tree):
        self.tree = tree
        self.tree.nodes.append(self)
        self.wc = tree.whether_continuous
        for child in self.children.values():
            if child is not None:
                child.feed_tree(tree)
    def predict_one(self, x):
        if self.category is not None:
            return self.category
        if self.is_continuous:
            if x[self.feature_dim] < self.tar:
                return self.left_child.predict_one(x)
            return self.right_child.predict_one(x)
        if self.is_cart:
            if x[self.feature_dim] == self.tar:
                return self.left_child.predict_one(x)
            return self.right_child.predict_one(x)
        else:
            try:
                return self.children[x[self.feature_dim]].predict_one(x)
            except KeyError:
                return self.get_category()
    def predict(self, x):
        return np.array([self.predict_one(xx) for xx in x])
    def view(self, indent=4):
        print("  " * indent * self._depth, self.info)
        for node in sorted(self.children):
            node = self.children[node]
            if node is not None:
                node.view()
    
    
class ID3Node(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = "ent"
class C45Node(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = "ratio"
class CartNode(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = "gini"
        self.is_cart = True