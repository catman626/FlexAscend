import matplotlib.pyplot as plt
from weightDistribution2D import parseCkpt, labutils

plt.rcParams["font.family"] = "SimHei"

def drawBar(data):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(data)), data)
    plt.xlabel("层号", fontsize=labutils.labelFontSize)
    plt.ylabel("权重最大值", fontsize=labutils.labelFontSize)
    plt.title("不同层 outProj 的最大值分布", fontsize=labutils.titleFontSize)
    plt.show()

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckptFile", )

    args = parser.parse_args()

    parsed = parseCkpt(args.ckptFile)

    outProjDistribution = [ l["outProj"] for l in parsed ]

    drawBar(outProjDistribution)

