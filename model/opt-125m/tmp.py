import mindspore as ms
import torch 

ckpt = ms.load_checkpoint("mindspore_model.ckpt") 
for n, w in ckpt.items():
    w = w.asnumpy()
