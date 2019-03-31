import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
    
from g_CNN.Networks import *
from Util.Util import DataUtil

def main():
    nn = NN()
    epoch = 10
    x, y = DataUtil.get_dataset("mnist", "../Data/mnist.txt", quantized=True, one_hot=True)
    nn.add("ReLU", (x.shape[1], 24))
    nn.add("ReLU", (24,))
    nn.add("CrossEntropy", (y.shape[1],))
    
    
    nn.fit(x, y, lr=0.001, epoch=epoch, train_rate=0.8, metrics=["acc"], record_period=1, verbose=2)
    
    
    