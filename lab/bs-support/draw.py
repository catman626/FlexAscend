import matplotlib.pyplot as plt
import numpy as np
# batch_sizes = ['16', '32', '64']
# methods = ['Vanilla', 'Compression']
# throughput = {
#     'vanilla': [120, 180, 210],  # throughput for batch sizes 16, 32, 64
#     'compression': [90, 150, 200]  # throughput for batch sizes 16, 32, 64
# }
plt.rcParams['font.family'] = "SimHei"

modelNames = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b"
]

methods = ['默认', '使用卸载']
bsSupported = {
    '默认': [2048, 1024, 256, 0, 0],  # throughput for batch sizes 16, 32, 64
    '使用卸载': [2048, 2048, 1024, 512, 256]  # throughput for batch sizes 16, 32, 64
}

x = np.arange(len(modelNames))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
for i, method in enumerate(bsSupported):
    ax.bar(x + (width * i),
            bsSupported[method], 
            width, 
            label=method)
    
    fails = [ xf for xf, yf in zip(x, bsSupported[method]) if yf == 0 ]
    
    label = "无法运行" if i == 0 else None
    ax.scatter(fails, [30] * len(fails), marker='x', color='red', label=label)

ax.set_xlabel('模型', fontsize=16)
ax.set_ylabel('支持的批大小', fontsize=16)
ax.set_title('最大支持的批大小', fontsize=20)
ax.set_xticks(x + width/2)
ax.set_xticklabels(modelNames)
ax.legend()

plt.show()

