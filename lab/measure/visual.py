
import re
import sys
sys.path.append("..")
from labutils import drawLineChart  

def extract(line):
    
    matched = re.match(r"^batch size: (\d+), time: ([\d]+.[\d]+) seconds",line)
    bs = int(matched.group(1))
    t = float(matched.group(2))
    
    return bs, t

def measure(logfile):
    """
    input : logfile
    output: a dict of {batchSize: time}
    """
    extracted = {}
    
    with open(logfile, 'r') as f:
        for line in f:
            bs, t = extract(line)
            extracted[bs] = t
    
    drawLineChart(extracted,title="bmm time")

    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", type=str)

    args = parser.parse_args()

    measure(args.logfile)