import matplotlib.pyplot as plt

# Example data (replace with yours)
batch_sizes = [1, 8, 16, 32, 64]
throughput = [100, 800, 1500, 2800, 3000]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(x=[ str(bs) for bs in batch_sizes] , height=throughput)

# Customize the chart
plt.xlabel('batch-size', fontsize=12)
plt.ylabel('throughput (ops/sec)', fontsize=12)
plt.title('throughput vs. batch Size', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()