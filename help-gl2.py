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

# Calculate means and standard deviations
means = np.array([np.mean(m) for m in measurements])
std_devs = np.array([np.std(m) for m in measurements])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(concentrations, means)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data points with error bars
plt.errorbar(concentrations, means, yerr=std_devs, fmt='o', color='black', 
             markersize=8, capsize=5, capthick=2, label='Data with std dev')

# Plot the linear fit
x_fit = np.linspace(0, 110, 100)
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, '--', color='black', linewidth=1.5, 
         label=f'Linear fit: y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}')

# Customize the plot
plt.title('Current Detection Linear Fitting', fontsize=14)
plt.xlabel('Concentration (nmol/L)', fontsize=12)
plt.ylabel('Average Current (μA)', fontsize=12)
plt.grid(linestyle='--', alpha=0.5)
plt.legend(fontsize=10)

# Set axis limits
plt.xlim(0, 110)
plt.ylim(0, 9)

# Add regression info to the plot
plt.text(70, 1.5, 
         f'Regression Equation:\ny = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}',
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()