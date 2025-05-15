import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# zhfont = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Normal.otf") 

plt.rcParams['font.family'] = 'SimHei'


def makeAutoPercent(values):
        def autoPercent(pct):
                total = sum(values)
                val = total * pct / 100
                return f"{val:.2f} GB\n({pct:.1f}%)"

        return autoPercent
# Data for the pie charts
labels = ['注意力层', '前馈层', '其他']
model1= [ 162.5669 , 218.2546 , 2.3190 ]
model2= [ 27.7552 , 37.5170 , 1.3477 ]
model3= [ 1.1408 , 1.5627 , 0.3845 ]
# colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
colors = ['#ff9999','#66b3ff','#99ff99']
explode = (0.05, 0.05, 0.05)  # Explode each slice a little

# Create a figure with 1 row and 3 columns
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# First pie chart
ax1.pie(model1, explode=explode, colors=colors,
        autopct=makeAutoPercent(model1), shadow=True, startangle=90)
ax1.set_title('OPT-175b')

# Second pie chart
ax2.pie(model2, explode=explode, colors=colors,
        autopct=makeAutoPercent(model2), shadow=True, startangle=90)
ax2.set_title('OPT-30b')

# Third pie chart
ax3.pie(model3, explode=explode, colors=colors,
        autopct=makeAutoPercent(model3), shadow=True, startangle=90)
ax3.set_title('OPT-1.3b')

# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
ax2.axis('equal')
ax3.axis('equal')

# Adjust layout to prevent overlapping
plt.tight_layout()

# plt.legend(labels, title="模型组件",  loc='upper right', prop=zhfont, title_fontproperties=zhfont)
plt.legend(labels, title="模型组件",  loc='upper right')

# Show the plot
plt.show()

