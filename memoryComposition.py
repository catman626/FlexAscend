import utils
from config import getOptConfig, OptConfig
from utils import GB

cfgs = {
    "opt-175b" : getOptConfig("opt-175b"), 
    "opt-30b" : getOptConfig("opt-30b"), 
    "opt-2.7b": getOptConfig("opt-2.7b"), 
    "opt-1.3b" : getOptConfig("opt-1.3b"), 
}


bs = 512
seqlen = 128

for cfg in cfgs.values():
    w, c, h = utils.model_bytes(cfg), \
        utils.cache_bytes(cfg, bs, seqlen), \
            utils.hidden_bytes(cfg, bs, seqlen)
    print(f"memory composition: model weight:{w/GB:.4f} GB"
          f", KV cache:{c/GB:.4f} GB,"
          f", hidden :{h/GB:.4f} GB")