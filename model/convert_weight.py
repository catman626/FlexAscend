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
    name = name.replace("model.decoder.", "")
    
    if not name.startswith("layers."):
        name = name.replace("embed_tokens.weight", "inputEmbed.tokenEmbedWeight")
        name = name.replace("embed_positions.weight", "inputEmbed.posEmbedWeight")
        name = name.replace("final_layer_norm.weight", "outputEmbed.norm.gamma")
        name = name.replace("final_layer_norm.bias", "outputEmbed.norm.beta")
        name = name.replace("lm_head.weight", "outputEmbed.tokenWeight")
    else :
        name = name.replace("self_attn_layer_norm.weight", "attn.attnLayerNorm.gamma")
        name = name.replace("self_attn_layer_norm.bias",   "attn.attnLayerNorm.beta")
        name = name.replace("final_layer_norm.weight", "ffn.layerNorm.gamma")
        name = name.replace("final_layer_norm.bias", "ffn.layerNorm.beta")

        name = name.replace("self_attn", "attn")
        name = name.replace("fc", "ffn.linear")
        name = name.replace("_proj", "Proj")

    return name
    
def main(torchCkpt, mindsporeCkpt):
    print(">>> load weight begin")
    torchWeight = torch.load(torchCkpt)
    print("<<< load weight finished")

    mindsporeWeight = []
    for name, weight in tqdm(torchWeight.items()):
        name = replaceName(name)
        
        mindsporeWeight.append({ 'name': name, 'data': torch2ms(weight)})

    ms.save_checkpoint(mindsporeWeight, mindsporeCkpt)


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("torchckpt", type=str)
    parser.add_argument("mindsporeckpt", type=str)

    args = parser.parse_args()

    main(args.torchckpt, args.mindsporeckpt)
