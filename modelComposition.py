import utils
from config import getOptConfig, OptConfig
from utils import GB


cfgs = {
    "opt-175b" : getOptConfig("opt-175b"), 
    "opt-30b" : getOptConfig("opt-30b"), 
    "opt-2.7b": getOptConfig("opt-2.7b"), 
    "opt-1.3b" : getOptConfig("opt-1.3b"), 
}

for c in cfgs.values():
    a, f, o = utils.model_components_bytes(c)
    print(f"model_components_bytes: attn:{a/GB:.4f} GB"
          f", ffn:{f/GB:.4f} GB,"
          f", other:{o/GB:.4f} GB")