import matplotlib.pyplot as plt

def plot_data(x, y):
    plt.plot(x, y, marker='o')
    plt.title('Sample Data Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()


def gen():
    while True:
        yield 1

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data from a file")
    parser.add_argument("inputfile", type=str)
    parser.add_argument("--outputfile", type=str, default="output.png", help="Output file for the plot")

    args = parser.parse_args()
