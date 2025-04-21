import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data
concentrations = np.array([1, 2, 4, 10, 20, 50, 100])
measurements = [
    [0.35, 0.34, 0.37],
    [0.52, 0.54, 0.50],
    [0.76, 0.82, 0.75],
    [1.50, 1.40, 1.60],
    [2.60, 2.32, 2.75],
    [6.40, 6.32, 6.56],
    [7.25, 6.83, 7.70]
]

# Calculate statistics
means = np.array([np.mean(m) for m in measurements])
variances = np.array([np.var(m) for m in measurements])
std_devs = np.array([np.std(m) for m in measurements])

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(concentrations, means)

# Create figure with two subplots
plt.figure(figsize=(14, 6))

# First subplot - Bar chart with variance
plt.subplot(1, 2, 1)
bars = plt.bar([str(c) for c in concentrations], means, 
              yerr=std_devs, capsize=5, 
              color='0.7', edgecolor='black')

# Add variance text
for i, (mean, var) in enumerate(zip(means, variances)):
    plt.text(i, mean + std_devs[i] + 0.2, f'σ²={var:.3f}', 
             ha='center', va='bottom', fontsize=9)

plt.title('concentration-signal relationship', fontsize=12)
plt.xlabel('Concentration (nmol/L)', fontsize=10)
plt.ylabel('Average Current (μA)', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Second subplot - Linear fitting
plt.subplot(1, 2, 2)
plt.errorbar(concentrations, means, yerr=std_devs, fmt='o', 
            color='black', markersize=6, capsize=5, label='Data')

# Regression line
x_fit = np.linspace(0, 110, 100)
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, '--', color='0.5', 
        label=f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}')

plt.title('Linear Regression', fontsize=12)
plt.xlabel('Concentration (nmol/L)', fontsize=10)
plt.ylabel('Average Current (μA)', fontsize=10)
plt.grid(linestyle='--', alpha=0.5)
plt.legend(fontsize=9)
plt.xlim(0, 110)
plt.ylim(0, 9)

# Adjust layout
plt.tight_layout()
plt.show()