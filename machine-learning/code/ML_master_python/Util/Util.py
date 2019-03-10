import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import pi, sqrt ,ceil



class DataUtil:
    naive_sets = {"mushroom", "balloon", "mnist", "cifar", "test"}
    
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