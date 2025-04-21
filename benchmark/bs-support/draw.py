import matplotlib.pyplot as plt
import numpy as np
# batch_sizes = ['16', '32', '64']
# methods = ['Vanilla', 'Compression']
# throughput = {
#     'vanilla': [120, 180, 210],  # throughput for batch sizes 16, 32, 64
#     'compression': [90, 150, 200]  # throughput for batch sizes 16, 32, 64
# }

modelNames = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b"
]

methods = ['vanilla', 'offload']
bsSupported = {
    'vanilla': [2048, 1024, 256, 0, 0],  # throughput for batch sizes 16, 32, 64
    'offload': [2048, 2048, 1024, 512, 256]  # throughput for batch sizes 16, 32, 64
}

x = np.arange(len(modelNames))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
for i, method in enumerate(bsSupported):
    ax.bar(x + (width * i),
            bsSupported[method], 
            width, 
            label=method)

ax.set_xlabel('Model')
ax.set_ylabel('batchSize supported')
ax.set_title('batch-size support Comparison by Batch Size and Compression Method')
ax.set_xticks(x + width/2)
ax.set_xticklabels(modelNames)
ax.legend()

plt.show()

