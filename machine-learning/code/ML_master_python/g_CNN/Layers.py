import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
import numpy as np
import tensorflow as tf
from math import ceil

from Util.Timing import Timing

#Abstract Layers
class Layer:
    LayerTiming = Timing
    
    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.is_fc = self.is_sub_layer = False
        self.apply_bias = kwargs.get("apply_bias", True)
    
    def __str__(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return str(self) #会调用__str__()
    
    @property
    def name(self):
        return str(self)
    
    @property
    def root(self):
        return self

    def derivative(self, y):
        pass
    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def activate(self, x, w, bias=None, predict=False):
        """
        如果是全连接层，则要将数据拉平
        如果是非全连接则，调用其定义的激活函数
        如果没有偏置，则调用其定义的激活函数，不加偏置参数
        否则调用自身的激活函数
        """
        if self.is_fc:
            fc_shape = np.prod(x.get_shape()[1:])
            x = tf.reshape(x, [-1, int(fc_shape)])
        if self.is_sub_layer:
            return self._activate(x, predict)
        if not self.apply_bias:
            return self._activate
        
        return self._activate(x.dot(w) + bias, predict)
    
    @LayerTiming.timeit(level=1, prefix="[Core] ")
 
    #定义求神经元输入的虚函数
    def _activate(self, x, predict):
        pass
      
class SubLayer(Layer):
    
    def __init__(self, parent, shape):
        Layer.__init__(self, shape)
        self.parent = parent
        self.description = ""
        self.is_sub_layer = True
    @property
    def root(self):
        root = self.parent
        while root.parent:
            root = root.parent
        return root
    @property
    def info(self):
        return "Layer  :  {:<16s} - {} {}".format(self.name, self.shape[1], self.description)

class ConvLayer(Layer):
    pass
class ConvPoolLayer(ConvLayer):
    pass
class ConvLayerMeta(type):
    def __new__(mcs, *args, **kwargs):
        """
        args:
        ('ConvTanh',
        (<class 'g_CNN.Layers.ConvLayer'>, <class 'g_CNN.Layers.Tanh'>),
        {'__module__': 'g_CNN.Layers', '__qualname__': 'ConvTanh'})
        """
        name, bases, attr = args[:3]
        conv_layer, layer = bases
        """
        元类的思想：
        conv_layer不同对应的激活函数也不同，我们将元类类比一个工厂，将不同的conv_layer 与激活函数组装起来
        形成对应的激活函数放入类的方法字典中，然后用type这个基类我们类比为模子，生成一个新的类
        将conv_layer,layer传进来
        
        """
        def __init__(self, shape, stride=1, padding="SAME"):
            conv_layer.__init__(self, shape, stride, padding)
        def _conv(self, x, w):
            return tf.nn.conv2d(x, w, strides=[1, self.stride, self.stride, 1], padding=self.pad_flag)
        def _activate(self, x, w, bias, predict):
            res = self._conv(x, w) + bias
            return layer._activate(self, res, predict)
        def activate(self, x, w, bias=None, predict=False):
            if self.pad_flag == "VALID" and self.padding > 0:
                _pad = [self.padding] * 2
                x = tf.pad(x,[[0,0], _pad, _pad,[0,0]], "CONSTANT")
            return _activate(self, x, w, bias, predict)
        for key, value in locals().itmes():
            if str(value).find("function") >= 0:
                attr[key] = value
        
        return type()
#Activation Layers

class Tanh(Layer):
    def _activate(self, x, predict):
        return tf.tanh(x)
class Sigmoid(Layer):
    def _activate(self, x, predict):
        return tf.nn.sigmoid(x)
class ELU(Layer):
    def _activate(self, x, predict):
        return tf.nn.elu(x)
class ReLU(Layer):
    def _activate(self, x, predict):
        return tf.nn.relu(x)
class Softplus(Layer):
    def _activate(self, x, predict):
        return tf.nn.softplus(x)
class Identical(Layer):
    def _activate(self, x, predict):
        return x

#Convolution Layers
class ConvTanh(ConvLayer, Tanh, metaclass=ConvLayerMeta):
    pass
class ConvSigmoid(ConvLayer, Tanh, metaclass=ConvLayerMeta):
    pass
class ConvELU(ConvLayer, ELU, metaclass=ConvLayerMeta):
    pass
class ConvSoftplus(ConvLayer, softplus, metaclass=ConvLayerMeta):
    pass
class ConvIdentical(ConvLayer, Identical, metaclass=ConvLayerMeta):
    pass

#Pooling Layers
class MaxPool(ConvPoolLayer):
    def _activate(self, x, *args):
        return tf.nn.max_pool
class AvgPool(ConvPoolLayer):
    def _activate(self, x, *args):
        return tf.nn.avg_pool
#Cost Layers
class CostLayer(Layer):
    def calculate(self, y, y_pred):
        return self._activate(y_pred, y)
class CrossEntropy(CostLayer):
    def _activate(self, x, y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y))
class MSE(CostLayer):
    def _activate(self, x, y):
        return tf.reduce_mean(tf.square(x - y))

#Special Layers
class Dropout(SubLayer):
    def __init__(self, parent, shape, drop_prob=0.5):
        pass
    def _activate(self, x, predict):
        pass
    

#Factory
class LayerFactory:
    available_root_layers = {
        #Normal Layers
        "Tanh": Tanh, "Sigmoid": Sigmoid,
        "ELU" : ELU, "ReLU": ReLU,
        "Softplus": Softplus,
        "Identical": Identical,
        
        #Cost Layers
        "CrossEntropy": CrossEntropy,
        "MSE": MSE,
        
        #Conv Layers
        "ConvTanh": ConvTanh,
        "ConvSigmoid": ConvSigmoid,
        "ConvELU": ConvELU,
        "ConvReLU": ConvReLU,
        "ConvSoftplus": ConvSoftplus,
        "ConvIdentical": ConvIdentical,
        "MaxPool": MaxPool, "AvgPool": AvgPool
    }
    
    available_special_layers = {
        "Dropout": Dropout,
        "Normalize": Normalize,
        "ConvDrop": ConvDrop,
        "ConvNorm": ConvNorm
    }
    special_layer_default_params = {
        "Dropout": (0.5,),
        "Normalize": ("Identical", 1e-8, 0.9),
        "ConvDrop": (0.5,),
        "ConvNorm": ("Identical", 1e-8, 0.9)
    }
    def get_root_layer_by_name(self, name, *args, **kwargs):
        if name not in self.available_special_layers:
            if name in self.available_root_layers:
                layer = self.available_root_layers[name]
            else:
                raise ValueError("Undefined layer '{}' found".format(name))
            return layer(*args, **kwargs)
        return None
    def get_layer_by_name(self, name, parent, current_dimension, *args, **kwargs):
        layer = self.get_root_layer_by_name(name, *args, **kwargs)
        if layer:
            return layer, None