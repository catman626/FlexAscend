import matplotlib.pyplot as plt
import numpy as np

batchSizes = [
    8,
    16,
    32,
    64,
    128
]

bsSupported = [
    0.0498,
    0.05,
    0.06,
    0.054,
    0.03
]


fig, ax = plt.subplots(figsize=(10, 6))
for i, method in enumerate(bsSupported):
    ax.bar([ str(bs) for bs in batchSizes],
           bsSupported)

ax.set_xlabel('batch size')
ax.set_ylabel('throughput')
ax.set_title('batch-size support Comparison by Batch Size and Compression Method')
ax.legend()

plt.show()

