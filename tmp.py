import numpy as np
import matplotlib.pyplot as plt

# Sample data - replace with your actual tensor values
num_layers = 12
num_components = 5
layer_numbers = np.arange(1, num_layers + 1)
component_names = [f'Component {i+1}' for i in range(num_components)]

# Generate random tensor values for demonstration (shape: layers × components)
np.random.seed(42)
tensor_values = np.random.rand(num_layers, num_components) * 100

# Create the plot
plt.figure(figsize=(10, 6))

# Create the heatmap with imshow
im = plt.imshow(tensor_values.T,  # Transpose to get components on y-axis
                cmap='viridis',
                aspect='auto',
                origin='lower',
                interpolation='nearest')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Tensor Value')

# Customize ticks and labels
plt.xticks(np.arange(num_layers), layer_numbers)
plt.yticks(np.arange(num_components), component_names)

# Add grid lines
# plt.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
# plt.gca().set_xticks(np.arange(-0.5, num_layers), minor=True)
# plt.gca().set_yticks(np.arange(-0.5, num_components), minor=True)

# Add labels and title
plt.xlabel('Layer Number')
plt.ylabel('Model Component')
plt.title('Model Component Values Across Layers')

# Add value annotations if desired (for smaller matrices)

plt.tight_layout()
plt.show()