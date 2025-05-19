import re
import matplotlib.pyplot as plt
import numpy as np

def parseLog(logfile):
    '''
    output 
    a list of record
    '''
    parsed = []
    with open(logfile, 'r') as f:
        banner = ">>>>>>>>>>>>>>>>>> OPT run <<<<<<<<<<<<<<<<<<"
        records = f.read().split(banner)
        records = [r for r in records if r.strip()]

        for r in records:
            bs = None
            thpt = None
            compress = None
            for l in r.split("\n"):
                matchedThpt = re.match(r" >>> throughput : (\d+.\d+) token/s", l)
                matchedBS = re.match(r" >>> batchSize: (\d+)", l)
                matchedCompress = re.match(r" >>> compress: (True|False)", l)

                if matchedThpt:
                    # print(" >>> matched!!! ")
                    thpt = matchedThpt.group(1)
                    thpt = float(thpt)
                if matchedBS:
                    bs = matchedBS.group(1)
                    bs = int(bs)
                if matchedCompress:
                    compress = matchedCompress.group(1)
                    if compress == "True":
                        compress = True
                    else:
                        compress = False

            parsed.append({
                "batchSize" : bs,
                "throughput": thpt,
                "compress" : compress
            })
            parsed
    
    return parsed

def drawBarsGroup2(parsedLog:dict, title="", xlabel="", ylabel=""):
    ''' 
    the parsedLog looks like:
    { 
        method1: { 
            8: 100,
            16: 200
            ...
        },
        method2: {}
    }'''
    modes = parsedLog.keys()
    
    batchSizes = list(list(parsedLog.values())[0].keys())
    batchSizes.sort()
   
    graphData = { m: [parsedLog[m][bs] for bs in batchSizes ] for m in modes }

    x = np.arange(len(batchSizes))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(modes):
        heights = np.array(graphData[m])
        ax.bar(x + (width * i),
                heights, 
                width, 
                label=m)
        
        for px , h in zip(x+(width * i), heights):
            ax.text(px, h , f"{h:.3f}", ha='center', va='bottom')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([ str(bs) for bs in batchSizes] )
    ax.legend()

    plt.show()