import matplotlib.pyplot as plt
import numpy as np

# Data
def drawCompressNew(report):
    # batch_sizes = ['16', '32', '64']
    # methods = ['Vanilla', 'Compression']
    # throughput = {
    #     'Vanilla': [120, 180, 210],  # throughput for batch sizes 16, 32, 64
    #     'Compression': [90, 150, 200]  # throughput for batch sizes 16, 32, 64
    # }

    x = np.arange(len(report["batchSizes"]))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for each method
    for i, method in enumerate(['vanilla', 'compression']):
        rects = ax.bar(x + (width * i), 
                    report[method], 
                    width, 
                    label=method)

    # Add labels, title and legend
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (requests/sec)')
    ax.set_title('Throughput Comparison by Batch Size and Compression Method')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(report["batchSizes"])
    ax.legend()

    # Add value labels on top of each bar
    for rect in ax.containers:
        ax.bar_label(rect, padding=3)

    plt.tight_layout()
    plt.show()

def parseRecord(record):
    compression = None
    batchSize = None
    throughput = None
    
    lines = record.split("\n")
    for l in lines:
        if "compress" in l:
            if "True" in l:
                compression = True
            elif "False" in l:
                compression = False
            else:
                raise ValueError("compression should be True or False")
                
        if "batchSize" in l:
            batchSize = int(l.split(":")[1].strip())
        if "throughput" in l:
            t = l.split(":")[1].strip().split(" ")[0]
            throughput = float(t)
        
    assert compression is not None \
        and batchSize is not None \
            and throughput is not None, \
                f"record is: \n {record}\n" 
        
    return batchSize, throughput, compression
        
def parseReport(reportFile):
    with open(reportFile, 'r') as f:
        content = f.read()
        records = [ r.strip() for r in content.split(">>>"*6) if r.strip() ]

    report = {
        "vanilla":      dict(),
        "compression":      dict(),
    }
    for r in records:
        bs, throughput, compress = parseRecord(r)
        if compress:
            report["compression"][bs] = throughput
        else:
            report["vanilla"][bs] = throughput

    throughputs = {
        'vanilla': [],
        'compression': [],
        'batchSizes': []
    }
    for bs in report["compression"]:
        if bs in report["vanilla"]:
            throughputs['vanilla'].append(report["vanilla"][bs])
            throughputs['compression'].append(report["compression"][bs])
            throughputs['batchSizes'].append(bs)
    
    return throughputs 
    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="python bar.py xxx.txt"
        " need to provide the record file")
    parser.add_argument("reportFile")

    args = parser.parse_args()

    parsedReport = parseReport(args.reportFile)
    
    drawCompressNew(parsedReport)
