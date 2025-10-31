import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import re
import sys

plt.rcParams["font.family"] = "SimHei"
# Sample data - replace this with your actual tensor values

sys.path.append("..")   
import labutils



def draw2DGrapgh(data):
    '''
    data in [layerno][component]
    '''
    nLayers = len(data)
    weightNames = [ "qProj", "kProj", "vProj", "outProj", "linear1", "linear2" ]

    value = np.array([ [ l[n] for n in weightNames ] for l in data ])
    print(value.shape)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Create the heatmap with imshow
    im = plt.imshow(value.T,  # Transpose to get components on y-axis
                    cmap='viridis',
                    aspect='auto',
                    origin='lower',
                    interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('权重最大值', fontsize=labutils.labelFontSize)

    # Customize ticks and labels
    plt.xticks(np.arange(nLayers), np.arange(1, nLayers+1))
    plt.yticks(np.arange(len(data[0])), data[0].keys())


    # Add labels and title
    plt.xlabel('层号', fontsize=labutils.labelFontSize)
    plt.ylabel('模型组件', fontsize=labutils.labelFontSize)
    plt.title('不同层的权重最大值分布', fontsize=labutils.titleFontSize)


    plt.tight_layout()
    plt.show()

def parseCkpt(ckptFile):
    data = torch.load(ckptFile)
    
    parsed = dict()
    for n, v in data.items():
        matched = re.match(r"^layers\.(\d+)\.(ffn|attn)\.([a-zA-Z0-9]+)\.(weight|bias)", n)
        if matched is not None:
            layerno = int(matched.group(1))
            name = matched.group(3)
            wob = matched.group(4)

            if layerno not in parsed:
                parsed[layerno] = dict()
            if name != "layernorm" and wob == "weight":
                assert name not in parsed[layerno], f"Duplicated weight name {name} in layer {layerno}"
                mx = torch.max(v).item()
                parsed[layerno][name] = mx
        else :
            print(f"Not matched: {n}")
                
    nLayers = len(parsed)
    listParsed = [ _ for _ in range(nLayers) ]
    for i in range(nLayers):
        listParsed[i] = parsed[i]
    # print to check
    for layerno, v in enumerate(listParsed):
        print(f"Layer {layerno}:")
        for name, value in v.items():
            print(f"  {name}: {value}")
    return listParsed
    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckptFile", type=str)

    args = parser.parse_args()
    
    weightDistribution = parseCkpt(args.ckptFile)
    draw2DGrapgh(weightDistribution)