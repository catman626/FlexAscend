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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("logfile",  )

    # args = parser.parse_args()
    
    # if args.logfile is not None:
    if False:
        parsed = labutils.parseLog(args.logfile)
        # print(parsed)
        throughputs = extract(parsed)
    else:
        throughputs = {
            "启用压缩": { 
                8: 2.413, 
                16: 4.4489, 
                32: 9.7164,
                64: 17.0082,
                128: 19.5259},
            "默认": { 
                8: 0.7942, 
                16: 1.5809, 
                32: 3.1156,
                64: 6.1398,
                128: 11.9354
                }
            
        }
   
    # print(throughputs)
    
    labutils.drawBarsGroup2(throughputs, 
                   title="压缩算法对吞吐量的影响",
                   xlabel="批大小",
                   ylabel="吞吐量")
    