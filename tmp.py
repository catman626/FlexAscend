import matplotlib.pyplot as plt
import numpy as np

# Data
batch_sizes = ['16', '32', '64']
methods     = ['Vanilla', 'Compression']
throughput  = {
    'Vanilla'       : [120, 180, 210],  # throughput for batch sizes 16, 32, 64
    'Compression'   : [90, 150, 200]  # throughput for batch sizes 16, 32, 64
}

x = np.arange(len(batch_sizes))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for each method
for i, method in enumerate(methods):
    rects = ax.bar(x + (width * i), 
                  throughput[method], 
                  width, 
                  label=method)

# Add labels, title and legend
ax.set_xlabel('Batch Size')
ax.set_ylabel('Throughput (requests/sec)')
ax.set_title('Throughput Comparison by Batch Size and Compression Method')
ax.set_xticks(x + width/2)
ax.set_xticklabels(batch_sizes)
ax.legend()

# Add value labels on top of each bar
for rect in ax.containers:
    ax.bar_label(rect, padding=3)

plt.tight_layout()
plt.show()