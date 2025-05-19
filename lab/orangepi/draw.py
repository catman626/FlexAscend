import matplotlib.pyplot as plt
import numpy as np
import re
import sys

sys.path.append("..")
import labutils

plt.rcParams['font.family'] = 'SimHei'
floatPattern = r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?"

def drawThroughputVSBS(throughputs:dict):
    """
    input : {
        bs0: t0,
        bs1: t1,
        ...
    }
    """

    print(" >>> throughputs: ", throughputs)
    batchSizes =  list(throughputs.keys())
    batchSizes.sort()
    throughputList = [throughputs[bs] for bs in batchSizes]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(batchSizes):
        ax.bar([ str(bs) for bs in batchSizes],
            throughputList)

    ax.set_xlabel('批大小')
    ax.set_ylabel('吞吐量')
    ax.set_title('香橙派上OPT-30b大模型吞吐量')

    plt.show()

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", type=str)

    args = parser.parse_args()
    
    parsed = labutils.parseLog(args.logfile)
    throughputs = { p["batchSize"] : p["throughput"] for p in parsed }
    
    drawThroughputVSBS(throughputs)
    