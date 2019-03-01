#encoding=utf-8
import numpy as np
import time
import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
from b_NavieBayes.Basic import *
from Util.Util import DataUtil
from Util.Timing import Timing
if __name__=="__main__":
    train_num = 6000
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset("mushroom","../Data/mushroom.txt", n_train=train_num,tar_idx=0)