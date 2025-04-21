import numpy as np
import matplotlib.pyplot as plt

# Data
concentrations = [1, 2, 4, 10, 20, 50, 100]
measurements = [
    [0.35, 0.34, 0.37],
    [0.52, 0.54, 0.50],
    [0.76, 0.82, 0.75],
    [1.50, 1.40, 1.60],
    [2.60, 2.32, 2.75],
    [6.40, 6.32, 6.56],
    [7.25, 6.83, 7.70]
]

# Calculate means and variances
means = [np.mean(m) for m in measurements]
variances = [np.var(m) for m in measurements]
std_devs = [np.std(m) for m in measurements]  # for error bars

# Create the plot
plt.figure(figsize=(10, 6))
bars = plt.bar([str(c) for c in concentrations], means, 
               yerr=std_devs, capsize=5, color="white", edgecolor='black')

# Add variance text above each bar
for i, (mean, var) in enumerate(zip(means, variances)):
    plt.text(i, mean + std_devs[i] + 0.2, f'σ²={var:.3f}', 
             ha='center', va='bottom', fontsize=9)

# Customize the plot
plt.title('Concentration-signal relationship', fontsize=14)
plt.xlabel('Concentration (nmol/L)', fontsize=12)
plt.ylabel('Average Current (μA)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()