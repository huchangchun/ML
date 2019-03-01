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
                    sample = sample.replace('"',"")
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