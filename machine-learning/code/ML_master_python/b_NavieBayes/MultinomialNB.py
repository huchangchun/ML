#encoding=utf-8
import numpy as np
import time

if __name__=="__main__":
    train_num = 6000
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset("mushroom","../../Data/mushroom.txt", n_train=train_num,tar_idx=0)