import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)
from f_NN.Networks import *
from f_NN.Layers import *
from Util.Util import DataUtil
def main():
    """
    模型的重点：
    1.构建网络时w,b的初始化:输入层，隐层..，输出层(损失层)
    w,b是层与层之间连接的参数，我们把参数的分层比较桥，
    第一个桥的w的维度是数据的(特征维度,第一个隐层的神经元个数)，b的维度就是(第一个隐层神经元个数)
    第二个桥的w的维度是(第一个隐层神经元的个数,第二个隐层的神经元的个数)
    ...
    最后一个桥是连接隐层和输出的，其维度是(最后一个隐层的神经元个数，输出层的类别）
    2.前向传导把所有激活值保存下来
    
    3.反向传播的梯度求解
    第一步有别与其他步，
    第一步调用CostLayers的bp_first进行bp算法的第一步得到损失层的输出对输入的梯度delta[-1]
    
    4.w,b的更新
    最后一步有别与前面的步骤，在更新w0的时候， W_(i-1)' = v^T_(i-1) * delta_(i)中，v用的是输入x_batch
   
    5.模型的优化
    采用了Adam的优化器，效果最稳定高效，在tf中也可以用这个
    
    6.模型的预测
    实际上是通过训练好的w,b一层层计算 X * W + b,并取最后一层输出作为预测值，然后取预测值中的最大值下标得到预测标签y^，
    最后通过y,y^求准确率
    """
    nn = NN()
    epoch = 1000
    x, y = DataUtil.gen_spiral(120, 7, 7, 4) #x(840,2) y(840,7):7是one_hot形式下的类别[1,0,0,0,0,0,0]表示第0个类别[0,1,0,0,0,0,0]表示第二个类别
    nn.add(ReLU((x.shape[1], 24)))
    nn.add(ReLU((24,)))
    nn.add(CostLayer((y.shape[1],),"CrossEntropy"))
    
    nn.fit(x, y, epoch=epoch, train_rate=0.8, metrics=["acc","f1-score"])
    nn.evaluate(x, y)
    nn.visualize2d(x, y)
    nn.show_timing_log()
    nn.draw_logs()
 
if __name__=="__main__":
    main()