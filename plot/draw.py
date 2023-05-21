import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
data = np.loadtxt('dragon.vis')

# 创建一个3D坐标轴对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图并调整点的大小
ax.scatter(data[:,0], data[:,1], data[:,2], s=5)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置初始视角
# ax.view_init(elev=10, azim=30)

# 显示图像并允许鼠标拖动查看不同角度
plt.show()
