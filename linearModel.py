import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 定义模型
def forward(x):
    return x * w + b


# 定义loss函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# 计算loss
w_list = []
b_list = []
mse_list = np.zeros((20, 20), dtype=float)
for i, w in enumerate(np.arange(0.0, 4.0, 0.2)):
    print("w=", w)
    for j, b in enumerate(np.arange(-2.0, 2.0, 0.2)):
        print("b=", b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print("\t", x_val, y_val, y_pred_val, loss_val)
        print("MSE=", l_sum / 3)
        mse_list[i][j] = l_sum / 3
        if w == 0:
            b_list.append(b)
    w_list.append(w)

# 绘制3D的loss曲线
x, y = np.meshgrid(w_list, b_list)
z = np.array(mse_list)
# 创建画布
fig = plt.figure()
# 创建3D坐标系
ax = Axes3D(fig)
# 3D平面图
ax.plot_surface(x, y, z, rstride=1,  # row行步长
                cstride=2,  # 列步长
                cmap="rainbow")  # 渐变颜色

plt.show()

"""
np.meshgrid(*xi, **kwargs)

Return coordinate matrices from coordinate vectors. 从坐标向量中返回坐标矩阵


直观的例子
二维坐标系中,X轴可以取三个值 1,2,3, Y轴可以取三个值 7,8, 请问可以获得多少个点的坐标?
显而易见是 6 个:
(1, 7) (2, 7) (3, 7)
(1, 8) (2, 8) (3, 8)

np.meshgrid() 就是干这个的!

#coding:utf-8
import numpy as np
# 坐标向量
a = np.array([1,2,3])
# 坐标向量
b = np.array([7,8])
# 从坐标向量中返回坐标矩阵
# 返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值
res = np.meshgrid(a,b)
#返回结果: [array([ [1,2,3] [1,2,3] ]), array([ [7,7,7] [8,8,8] ])]
"""

"""
3D立体图形
绘制三维图像主要通过 mplot3d 模块实现。
散点图 曲线图 平面图

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib notebook
3D绘图
3D绘图与2D绘图使用的方法基本一致，不同的是，操作的对象变为了 Axes3D() 对象。

x = [1,2,3,4]
y = [1,2,3,4]
X, Y = np.meshgrid(x, y)
 
# 创建画布
fig = plt.figure()
# 创建3D坐标系
ax = Axes3D(fig)
 
ax.plot_surface(X,
                Y,
                Z=X+Y
               )
"""
