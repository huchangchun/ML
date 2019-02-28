#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

x,y = [],[]
for sample in open("../Data/prices.txt","r"):
    xx, yy = sample.strip("\n").split(",") #'1985,299900\n'
    x.append(float(xx))
    y.append(float(yy))
x ,y = np.array(x), np.array(y)
#perform normalization 标准化
x = (x-x.mean()) /x.std()

#Scatter dataset
plt.figure()
plt.scatter(x,y,c='g',s=20)
plt.show()

x0 = np.linspace(-2, 4, 100)

#Get regression model under LSE criterion with degree 'deg'
#多项式 ，deg表示多项式的次数
#根据输入x,返回相应的预测的y
#fun = lambda x,y: x*y
#fun(2,3)=6
def get_model(deg):
    #ployfit(x,y,deg)会训练会返回使得的均方误差最小的参数p,亦即多项式的系数
    #polyval(p,x)根据多项式的系数和输入p,x预测返回的y值
    return lambda input_x = x0: np.polyval(np.polyfit(x, y, deg),input_x)

#给定x,y求损失
def get_cost(deg, input_x, input_y):
    #(y-y^)**2
    return 0.5 * ((get_model(deg)(input_x) - input_y) ** 2).sum()

test_set = (1,2,3,4)
for d in test_set:
    print(get_cost(d, x, y))

#Visualize results 
plt.scatter(x,y,c='g',s=20)
for d in test_set:
    plt.plot(x0, get_model(d)(), label="degree = {}".format(d))
plt.xlim(-2, 4) #将横轴限制在-2,4
plt.ylim(1e5, 8e5) #将纵轴限制在（10^5，8*10^5)
plt.legend() #让对应的label正确显示
plt.show()
#可以看到当n=4时，已经出现过拟合