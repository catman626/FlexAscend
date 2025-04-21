import matplotlib.pyplot as plt
import numpy as np

# Data
compression_methods = ['vanilla', 'compression']
batch_sizes = [16, 32, 64]
throughput = {
    'vanilla': [120, 180, 210],  # throughput for batch sizes 16, 32, 64
    'compression': [90, 150, 200]  # throughput for batch sizes 16, 32, 64
}



def drawCompress(report):
    """
    input: like this:
        throughputs = {
            'batchSizes':   []
            'vanilla':      [],
            'compression':  [],
        }
    """

    x = np.arange(2)  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for each batch size
    for i, bs in enumerate(report["batchSizes"]):
        offset = width * i - width  # centering the groups
        rects = ax.bar(x + offset, 
                    [report['vanilla'][i], report['compression'][i]], 
                    width, 
                    label=f'Batch Size {bs}')

    # Add labels, title and legend
    ax.set_xlabel('Compression Method')
    ax.set_ylabel('Throughput (token/s')
    ax.set_title('Throughput Comparison by Compression Method and Batch Size')
    ax.set_xticks(x)
    ax.set_xticklabels(compression_methods)
    ax.legend()

    # Add value labels on top of each bar
    for rect in ax.containers:
        ax.bar_label(rect, padding=3)

    plt.tight_layout()
    plt.show()