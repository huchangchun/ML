#encoding=utf-8
import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import cv2
from copy import deepcopy
from c_CvDTree.Node import *
from Util.Timing import Timing
from Util.Bases import ClassifierBase

class CvDBase(ClassifierBase):
    CvDBaseTiming = Timing()
    def __init__(self, whether_continuous=None, max_depth=None, node=None, **kwargs):
        super(CvDBase, self).__init__(**kwargs)
        """
        self.nodes :记录所有Node的列表
        self.layers ： 主要用于CART剪枝的属性
        self.roots  
        self.max_depth  ：决策树的最大树深
        self.root  ：根节点
        self.feature_sets 记录可选特征维度的列表
        self.prune_alpha 
        self.y_transformer = None
        self.wheter_continuous = whether_continuous
        """
        self.nodes = []
        self.layers = []
        self.roots = []
        self.max_depth = max_depth
        self.root = node
        self.feature_sets = []
        self.prune_alpha = 1
        self.y_transformer = None
        self.wheter_continuous = whether_continuous
        
        self._params["alpha"] = kwargs.get("alpha", None)
        self._params["eps"] = kwargs.get("eps", 1e-8)
        self._params["cv_rate"] = kwargs.get("cv_rate", False)
        self._params["train_only"] = kwargs.get("train_only", False)
        self._params["feature_bound"] = kwargs.get("feature_bound", None)
    @CvDBaseTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, alpha=None, eps=None,
            cv_rate=None, train_only=None, feature_bound=None):
        """
        1.初始化各种参数
        2.将y转换为数值型数据，np.unique()
        3.初始化树根self.root
        4.向根节点输入数据
            根据信息增益的度量，选择数据的某个特征来把数据划分成互不相交的好几份并分别喂给一个新的Node
        5.剪枝
        """
        if sample_weight is None:
            
        
class CvDMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        _, _node = bases
        
        def __init__(self, whether_continuous=None, max_depth=None, node=None, **_kwargs):
            tmp_node = node if isinstance(node, CvDNode) else _node
            CvDBase.__init__(self, whether_continuous, max_depth, tmp_node(**_kwargs))
            self._name = name
        attr["__init__"] = __init__
        return type(name, bases, attr)

class CartTree(CvDBase, CartNode, metaclass=CvDMeta):
    pass