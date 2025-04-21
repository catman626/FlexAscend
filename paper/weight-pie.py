
import matplotlib.pyplot as plt

def drawPie3(a, b, c):
    labels = ['weight', 'KV-cache', 'activation']
    sizes = [a, b, c]  # Values for each category
    colors = [
        '#D4B8FF',  # Soft purple (pastel lavender)
        '#FFF6A5',  # Faint yellow
        '#A5FFD6',  # Mint green
    ]

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

    # Equal aspect ratio ensures the pie is circular
    plt.axis('equal')  

    # Add title
    plt.title("distribution of memory usage")

    # Show plot
    plt.show()

    
if __name__ == "__main__":
    # Data
    modelSize = 2.443 
    cacheSize = 0.398 
    hiddenSize = 0.008
    totalSize = modelSize + cacheSize + hiddenSize


    drawPie3(modelSize/totalSize, cacheSize/totalSize, hiddenSize/totalSize)
