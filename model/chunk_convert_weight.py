from tqdm import tqdm
import torch
import mindspore as ms
import argparse

def torch2ms(value: torch.Tensor) -> ms.Tensor:
    """
    convert torch.Tensor to ms.Tensor with specified dtype
    """
    np_value = value.detach().numpy()
    return ms.Tensor(np_value)

def replaceName(name:str):
    name = name.replace("model.", "")
    name = name.replace("decoder.", "")
    
    if not name.startswith("layers."):
        name = name.replace("embed_tokens.weight", "inputEmbed.tokenWeight")
        name = name.replace("embed_positions.weight", "inputEmbed.posWeight")
        name = name.replace("final_layer_norm.weight", "outputEmbed.layernorm.weight")
        name = name.replace("final_layer_norm.bias", "outputEmbed.layernorm.bias")
        name = name.replace("lm_head.weight", "outputEmbed.tokenWeight")
    else :
        name = name.replace("self_attn_layer_norm.weight", "attn.layernorm.weight")
        name = name.replace("self_attn_layer_norm.bias",   "attn.layernorm.bias")
        name = name.replace("final_layer_norm.weight", "ffn.layernorm.weight")
        name = name.replace("final_layer_norm.bias", "ffn.layernorm.bias")

        name = name.replace("self_attn", "attn")
        name = name.replace("fc", "ffn.linear")
        name = name.replace("_proj", "Proj")
    return name
    
def convertFile(torchCkpt, mindsporeCkpt):
    print(">>> load weight begin")
    torchWeight :dict = torch.load(torchCkpt)
    print("<<< load weight finished")

    mindsporeWeight = []

    while torchWeight:
        torchname, weight =torchWeight.popitem()
        if torchname == "lm_head.weight":
            continue
        
        msname = replaceName(torchname)
        msdata = torch2ms(weight)
        
        mindsporeWeight.append({ 'name': msname, 'data': msdata})

        if msname == "inputEmbed.tokenWeight":
            mindsporeWeight.append({ "name": "outputEmbed.tokenWeight", "data": msdata})

        print("\t >>> convert item success")

    ms.save_checkpoint(mindsporeWeight, mindsporeCkpt)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("weightFiles", nargs="+", help="list all files needed to convert, and add the output prefix finally")

    args = parser.parse_args()

    pyckpts = args.weightFiles[:-1]
    msCkptPrefix = args.weightFiles[-1]
    
    print(f" >>> files to convert: ")
    for p in pyckpts:
        print(f"\t >>> {p}")
    
    
    print(f" >>> convert begin")
    for i, p in enumerate(pyckpts):
        outputFile = msCkptPrefix + f".{i}.ckpt"
        print(f" \t>>> convert begin: {p} to {outputFile}")
        convertFile(p, outputFile)
        print(f" \t<<< convert file end")
    
    print(f" <<< convert end")

