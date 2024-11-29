from mindspore import load_checkpoint, save_checkpoint


file = "model-weight/mindspore-weight.ckpt"
outputfile = "model-weight/mindspore-weight.1.ckpt"


weights = load_checkpoint(file)



modified = []

for name, weight in weights.items():
    name = name.replace("tokenEmbedWeight", "inputEmbed.tokenEmbedWeight")
    name = name.replace("posEmbedWeight", "inputEmbed.posEmbedWeight")
    name = name.replace("output_embed_layer_norm.weight", "outputEmbed.norm.gamma")
    name = name.replace("output_embed_layer_norm.bias", "outputEmbed.norm.beta")
    name = name.replace("lm_head.weight", "outputEmbed.tokenWeight")
    
    modified.append({ "name": name, "data": weight})

save_checkpoint(modified, outputfile)