#encoding=utf-8
import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
import time

from c_CvDTree.Tree import *
from Util.Util import DataUtil

def main(visualize=True):
    x, y = DataUtil.get_dataset("balloon.0(en)","../Data/balloon1.0(en).txt")
    fit_time = time.time()
    tree = CartTree(whether_continuous=[False] * 4)
    tree.fit(x, y, train_only=True)
    fit_time = fit_time - time.time()
    if visulaize:
        tree.view()
    estimate_time = time.time()
    tree.evaluate(x, y)
    estimate_time = estimate_time - time.time()
    print("Model building  :{:12.6} s\n"
          "Estimation      :{:12.6} s\n"
          "Total           :{:12.6} s\n".format(
                fit_time, estimate_time,
                fit_time + estimate_time
            )
    )
if __name__=="__main__":
    main(False)