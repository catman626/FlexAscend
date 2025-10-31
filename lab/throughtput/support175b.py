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
    width = 0.7  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(batchSizes):
        ax.bar([ str(bs) for bs in batchSizes],
            throughputList, width=width)

        ax.text(i , throughputs[method], throughputs[method], ha='center', va='bottom', fontsize=14) 
            # ax.text(px, h , f"{h:.3f}", ha='center', va='bottom')

    ax.set_xlabel('批大小', fontsize=labutils.labelFontSize)
    ax.set_ylabel('吞吐量', fontsize=labutils.labelFontSize)
    ax.set_title('OPT-175b大模型吞吐量',fontsize=labutils.titleFontSize)
    # ax.legend()

    plt.show()

def parseLog(logfile):
    throughputs = {}
    with open(logfile, 'r') as f:
        banner = ">>>>>>>>>>>>>>>>>> OPT run <<<<<<<<<<<<<<<<<<"
        records = f.read().split(banner)
        records = [r for r in records if r.strip()]
        for r in records:
            bs = None
            throughput = None
            for l in r.split("\n"):
                matchedThpt = re.match(r" >>> throughput : (\d+.\d+) token/s", l)
                matchedBS = re.match(r" >>> batchSize: (\d+)", l)
                if matchedThpt:
                    throughput = matchedThpt.group(1)
                    throughput = float(throughput)
                if matchedBS:
                    bs = matchedBS.group(1)
                    bs = int(bs)

            print(f" >>> bs: {bs}, throughput: {throughput}")
            throughputs[bs] = throughput
    
    return throughputs

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", type=str)

    args = parser.parse_args()
    
    throughputs = parseLog(args.logfile)
    drawThroughputVSBS(throughputs)
    