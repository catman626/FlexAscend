import matplotlib.pyplot as plt
percentLabelSize = 14
titleSize = 20

plt.rcParams['font.family'] = 'SimHei'

def makeAutoPercent(values):
    def autoPercent(pct):
        total = sum(values)
        val = total * pct / 100
        return f"{val:.2f} GB\n({pct:.1f}%)"

    return autoPercent

def draw3Pies():
    # Data for the pie charts
    labels = ['权重', 'KV缓存', '激活']
    models = {
        "OPT-175b" :[ 650.3498 , 576.0000 , 3.0000 ],
        "OPT-30b" :[ 111.6067 , 168.0000 , 1.7500 ],
        "OPT-1.3b" :[ 4.8857 , 24.0000 , 0.5000 ] 
    }

    # colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    colors = ['#ff9999','#66b3ff','#99ff99']
    explode = (0.05, 0.05, 0.05)  # Explode each slice a little

    # Create a figure with 1 row and 3 columns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    # Third pie chart


    for (mn, m), ax in zip( models.items(), [ax1, ax2, ax3]):
        ax.pie(m, explode=explode, colors=colors,
                autopct=makeAutoPercent(m), shadow=True, startangle=90, textprops={'fontsize': percentLabelSize})
        ax.set_title(mn, fontsize=titleSize)
            

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    ax2.axis('equal')
    ax3.axis('equal')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    plt.legend(labels, title="推理数据",  loc='upper right')


    # Show the plot
    plt.show()

if __name__ == "__main__":
    draw3Pies()