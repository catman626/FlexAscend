import matplotlib.pyplot as plt
import numpy as np

def makeAutoPercent(values):
        def autoPercent(pct):
                total = sum(values)
                val = total * pct / 100
                return f"{val:.2f} GB\n({pct:.1f}%)"

        return autoPercent
# Data for the pie charts
labels = ['weight', 'KV-cache', 'hidden']
model1 = [ 650.3498 , 576.0000 , 3.0000 ]
model2 = [ 111.6067 , 168.0000 , 1.7500 ]
model3 = [ 4.8857 , 24.0000 , 0.5000 ]
# colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
colors = ['#ff9999','#66b3ff','#99ff99']
explode = (0.05, 0.05, 0.05)  # Explode each slice a little

# Create a figure with 1 row and 3 columns
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# First pie chart
ax1.pie(model1, explode=explode, colors=colors,
        autopct=makeAutoPercent(model1), shadow=True, startangle=90)
ax1.set_title('opt-175b')

# Second pie chart
ax2.pie(model2, explode=explode, colors=colors,
        autopct=makeAutoPercent(model2), shadow=True, startangle=90)
ax2.set_title('opt-30b')

# Third pie chart
ax3.pie(model3, explode=explode, colors=colors,
        autopct=makeAutoPercent(model3), shadow=True, startangle=90)
ax3.set_title('opt-1.3b')

# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
ax2.axis('equal')
ax3.axis('equal')

# Adjust layout to prevent overlapping
plt.tight_layout()

plt.legend(labels, title="model components",  loc='upper right')

# Show the plot
plt.show()