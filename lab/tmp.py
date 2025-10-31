import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建数据
x = np.linspace(0, 10, 20)  # 从0到10生成20个均匀分布的点
y = np.sin(x)  # 计算每个x点的正弦值
y2 = np.cos(x)  # 另一组数据

# 创建图形和坐标轴
plt.figure(figsize=(8, 5))  # 设置图形大小

# 绘制折线图
plt.plot(x, y, label='正弦曲线', color='blue', marker='o', linestyle='-', linewidth=2)
plt.plot(x, y2, label='余弦曲线', color='red', marker='s', linestyle='--', linewidth=2)

# 添加标题和标签
plt.title('正弦和余弦函数曲线', fontsize=16)
plt.xlabel('X轴', fontsize=12)
plt.ylabel('Y轴', fontsize=12)

# 添加图例
plt.legend(fontsize=10)

# 设置网格
plt.grid(True, linestyle=':', alpha=0.7)

# 设置坐标轴范围
plt.xlim(0, 10)
plt.ylim(-1.2, 1.2)

# 显示图形
plt.tight_layout()  # 自动调整子图参数
plt.show()