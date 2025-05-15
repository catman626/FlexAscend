import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Sample data - replace this with your actual tensor values
num_layers = 12
num_components = 5
layer_numbers = np.arange(1, num_layers + 1)
component_names = [f'Component {i+1}' for i in range(num_components)]

# Generate random tensor values for demonstration (shape: layers × components)
np.random.seed(42)
tensor_values = np.random.rand(num_layers, num_components) * 100

# Create the plot
plt.figure(figsize=(10, 6))

# Create a heatmap
heatmap = plt.pcolor(tensor_values.T, cmap='viridis', linewidths=1)

# Add colorbar
cbar = plt.colorbar(heatmap)
cbar.set_label('Tensor Value')

# Set ticks and labels
plt.xticks(np.arange(0.5, num_layers + 0.5), layer_numbers)
plt.yticks(np.arange(0.5, num_components + 0.5), component_names)

# Add labels
plt.xlabel('Layer Number')
plt.ylabel('Model Component')
plt.title('Model Component Values Across Layers')

# Show plot
plt.tight_layout()
plt.show()