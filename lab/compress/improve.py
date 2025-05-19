import matplotlib.pyplot as plt
import numpy as np
import sys

plt.rcParams['font.family'] = "SimHei"

sys.path.append("..")   
import labutils

def extract(parsed):
    """
    input : [
        { 
        compress: True, 
        bs: xxx, 
        thpt: xxx },
        { }    
    ]    
    output should be: 
    
    extracted: [mode][bs][throughputs]
    """
    extracted = {}
    for r in parsed:
        bs = r["batchSize"]
        thpt = r["throughput"]
        mode = "启用压缩" if r["compress"]  else"默认"
        if mode not in extracted:
            extracted[mode] = {}
            
        assert bs not in extracted[mode], f"batchSize {bs} already exists in {mode}"
        extracted[mode][bs] = thpt

    return extracted

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", type=str, help="log file")

    args = parser.parse_args()
    
    parsed = labutils.parseLog(args.logfile)
    # print(parsed)
    throughputs = extract(parsed)
    
    # print(throughputs)
    labutils.drawBarsGroup2(throughputs, 
                   title="压缩算法对吞吐量的影响",
                   xlabel="批大小",
                   ylabel="吞吐量")
    