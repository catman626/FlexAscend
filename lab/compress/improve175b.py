import matplotlib.pyplot as plt
import numpy as np

mode = [
    "vanilla",
    "compress",
]
batchSizes = [
    8,
    16,
    32,
    64,
    128
]

throughputs = { 
    "vanilla": [
        0.0498, 
        0.05,
        0.06,
        0.054,
        0.03
    ], 
    "compress": [
        2 * 0.0498, 
        2 * 0.05,
        2 * 0.06,
        2 * 0.054,
        2 * 0.03
    ]
}
    
x = np.arange(len(batchSizes))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
for i, method in enumerate(mode):
    ax.bar(x + (width * i),
            throughputs[method], 
            width, 
            label=method)

ax.set_xlabel('batch-size')
ax.set_ylabel('throughput')
ax.set_title('throughput of OPT-175b')
ax.set_xticks(x + width/2)
ax.set_xticklabels([ str(bs) for bs in batchSizes] )
ax.legend()

plt.show()

