#encoding=utf-8
import matplotlib.pyplot as plt
import matplotlib
#设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False

label_list=['2014','2015','2016','2017'] #横坐标刻度显示
num_for_bar1 = [20,30,15,35] #纵坐标值1
num_for_bar2 = [15,30,40,20] #纵坐标值2
x = range(len(num_for_bar1))

"""
回执条形图
left:长条中点横坐标
height:长条高度
width:长条宽度
label:为设置legend准备
"""
rect1 = plt.bar(left=x, height=num_for_bar1, width=0.4, alpha=0.8, color='red', label='A')
rect2 = plt.bar(left=[i + 0.4 for i in x], height=num_for_bar2, width=0.4, alpha=0.8, color='green', label='B')
plt.ylim(0, 50) #y轴取值范围
plt.ylabel('数量')

"""
设置x轴刻度显示值
参数一：点坐标
参数二:显示值
"""

plt.xticks([index + 0.2 for index in x], label_list)
plt.xlabel("年份")
plt.title("GDP")
plt.legend() #设置题注

for rect in rect1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() /2 ,height+1, str(height), ha='center', va="bottom")
for rect in rect2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() /2 ,height+1, str(height), ha='center', va="bottom")
plt.show()
