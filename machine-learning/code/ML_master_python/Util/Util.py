import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import pi, sqrt ,ceil

class Util:
    @staticmethod
    def callable(obj):
        _str_obj = str(obj)
        if callable(obj):
            return True
        if "<" not in _str_obj and ">" not in _str_obj:
            return False
        if _str_obj.find("function") >= 0 or _str_obj.find("staticmethod") >= 0:
            return True

    @staticmethod
    def freeze_graph(sess, ckpt, output):
        print("Loading checkpoint...")
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
        print("Writing graph...")
        if not os.path.isdir("_Cache"):
            os.makedirs("_Cache")
        _dir = os.path.join("_Cache", "Model")
        saver.save(sess, _dir)
        graph_io.write_graph(sess.graph, "_Cache", "Model.pb", False)
        print("Freezing graph...")
        freeze_graph.freeze_graph(
            os.path.join("_Cache", "Model.pb"),
            "", True, os.path.join("_Cache", "Model"),
            output, "save/restore_all", "save/Const:0", "Frozen.pb", True, ""
        )
        print("Done")

    @staticmethod
    def load_frozen_graph(graph_dir, fix_nodes=True, entry=None, output=None):
        with gfile.FastGFile(graph_dir, "rb") as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            if fix_nodes:
                for node in graph_def.node:
                    if node.op == 'RefSwitch':
                        node.op = 'Switch'
                        for index in range(len(node.input)):
                            if 'moving_' in node.input[index]:
                                node.input[index] = node.input[index] + '/read'
                    elif node.op == 'AssignSub':
                        node.op = 'Sub'
                        if 'use_locking' in node.attr:
                            del node.attr['use_locking']
            tf.import_graph_def(graph_def, name="")
            if entry is not None:
                entry = tf.get_default_graph().get_tensor_by_name(entry)
            if output is not None:
                output = tf.get_default_graph().get_tensor_by_name(output)
            return entry, output

class DataUtil:
    naive_sets = {"mushroom", "balloon", "mnist", "cifar", "test"}
    @staticmethod
    def gen_xor(size=100, scale=1, one_hot=True):
        x = np.random.randn(size) * scale
        y = np.random.randn(size) * scale
        z = np.zeros((size, 2))
        z[x * y >= 0, :] = [0, 1]
        z[x * y < 0, :] = [1, 0]
        if one_hot:
            return np.c_[x, y].astype(np.float32), z
        return np.c_[x, y].astype(np.float32), np.argmax(z, axis=1)
    
    @staticmethod
    def is_naive(name):
        for naive_dataset in DataUtil.naive_sets:
            if naive_dataset in name:
                return True
        return False
    
    @staticmethod
    def get_dataset(name, path, n_train=None, tar_idx=None, shuffle=True,
                    quantize=False, quantized=False, one_hot=False, **kwargs):
        
        x = []
        with open(path,"r", encoding="utf8") as file:
            if DataUtil.is_naive(name):
                for sample in file:
                    x.append(sample.strip().split(","))
            elif name == "bank1.0":
                for sample in file:
                    sample = sample.replace('"',"")#双引号替换为空
                    x.append(list(map(lambda c: c.strip(), sample.split(";"))))
            else:
                raise NotImplementedError
        if shuffle:
            np.random.shuffle(x)
        tar_idx = -1 if tar_idx is None else tar_idx
        y = np.array([xx.pop(tar_idx) for xx in x])
        if quantized:
            x = np.asarray(x, dtype=np.float32)
            y = y.astype(np.int8)
            if one_hot:
                y = (y[...,None]==np.arange(np.max(y) + 1))
        else:
            x = np.asarray(x)
        if quantized or not quantize:
            if n_train is None:
                return x,y
            return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])
        x, y, wc, features, feat_dicts, label_dict = DataUtil.quantize_data(x,y,**kwargs)
        
        if one_hot:
            y = (y[...,None]==np.arange(np.max(y) + 1)).astype(np.int8)
        if n_train is None:
            return x, y ,wc, features, feat_dicts, label_dict
        return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:]),wc,features,feat_dicts,label_dict
   
    @staticmethod
    def quantize_data(x, y, wc=None, continuous_rate=0.1, separate=False):
        if isinstance(x, list):
            xt = map(list, zip(*x))
        else:
            xt = x.T
        features = [set(feat) for feat in xt] #将获取每个维度的特征的种类
        if wc is None:
            wc = np.array([len(feat) >= int(continuous_rate * len(y)) for feat in features])
        else:
            wc = np.asarray(wc)        
        feat_dicts = [
        {_l:i for i, _l in enumerate(feats)} if not wc[i] else None for i,feats in enumerate(features) #如果是连续型特征则设置为None
        ]
        if not separate:
            if np.all(~wc):
                dtype=np.int
            else:
                dtype =np.float32
            x = np.array([[feat_dicts[i][_l] if not wc[i] else _l for i,_l in enumerate(sample)] for sample in x], dtype=dtype)
            
        else:
            x = np.array([[feat_dicts[i][_l] if not wc[i] else _l for i, _l in enumerate(sample)] for sample in x], dtype=np.float32)
            x = (x[:, ~wc].astype(np.int), x[:, wc]) #将x按离散和连续特征划分
        
        label_dict = {l: i for i ,l in enumerate(set(y))}
        y = np.array([label_dict[yy] for yy in y], dtype=np.int8)
        label_dict ={i: l for l, i in label_dict.items()}
        return x, y, wc, features, feat_dicts, label_dict