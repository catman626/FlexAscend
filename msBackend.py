import numpy as np
import os
import mindspore as ms
from msCompress import compress, decompress
from mindspore import Tensor
from mindspore import dtype, ops

class AscendTensor:
    def __init__(self, name:str):
        self.val = None
    
    def store(self, data:Tensor):
        self.val = data
        
    def data(self):
        return self.val
        
class CPUTensor:
    def __init__(self, name:str):
        self.val = None
        self.cached = None
        self.name = name
        
    def store(self, data:Tensor):
        self.val = data

    def data(self):
        if self.cached is not None:
            cached = self.cached
            self.cached = None
            return cached
        
        return self.val

    def load(self, dstDev):
        assert dstDev in ["Ascend", "cpu"]

        if dstDev == "Ascend":
            self.cached = self.val.move_to("Ascend") 
    
class DiskTensor:
    weightHome = "weightHome"
    compress = False
    def __init__(self, name:str):
        self.name = name
        self.filename = None
        self.cached = None
        
    def store(self, data:Tensor):
        assert isinstance(data, Tensor)
        self.filename = os.path.join(DiskTensor.weightHome, self.name) + ".npy"
        if not os.path.exists(DiskTensor.weightHome):
            os.mkdir(DiskTensor.weightHome)

        self.shape = data.shape

        if DiskTensor.compress:
            data, extra = compress(data)
            
            np.save(data, self.filename)
            np.save(extra, self.filename+".extra")
        else:
            np.save(data, self.filename)

    def load(self, dstDev):
        assert dstDev in ["Ascend", "cpu"]
        t = np.load(self.filename)      # to np array
        t = Tensor(t)                   # to ms tensor
        
        if DiskTensor.compress:
            extra = np.load(self.filename + ".extra") 

            self.cached = decompress(t, extra, self.shape)
        else:
            self.cached = t

        if dstDev == "Ascend":
            self.cached = self.cached.move_to("Ascend")
        else:
            self.cached = self.cached.move_to("cpu")

    def data(self):
        if self.cached is not None: 
            cached = self.cached
            self.cached = None
            return cached 

        raise RuntimeError(f"disk-tensor fetch before load")

    @staticmethod
    def clear():
        for f in os.listdir(DiskTensor.weightHome):
            os.remove(os.path.join(DiskTensor.weightHome, f))
        
class FlexTensor:
    def __init__(self, name, shape=None, home:str="DISK"):
        self.shape = shape
        self.name = name
        self.home = home    # where the data is stored, Ascend or CPU, DISK
        
        if self.home == "Ascend":
            self.tensorCls = AscendTensor
        elif self.home == "cpu":
            self.tensorCls = CPUTensor
        elif self.home == "DISK":
            self.tensorCls = DiskTensor
        else:
            raise NotImplementedError(f"not implemented home: {self.home}")

        self.tensor = self.tensorCls(self.name)

    def store(self, data:Tensor):
        self.tensor.store(data)

    def load(self, dstDev):
        self.tensor.load(dstDev)

    def data(self):
        return self.tensor.data()

    def initZeros(self):
        return self.tensor.store(Tensor(np.zeros(self.shape, dtype=np.float32)))

    @staticmethod
    def setCompress(compress:bool):
        DiskTensor.compress = compress

    @staticmethod
    def clear():
        DiskTensor.clear()
        
def layernorm(x:FlexTensor, normalizedShape:FlexTensor, weight:FlexTensor, bias:FlexTensor):
    return ops.layer_norm(x.data(), normalizedShape, weight.data(), bias.data())

def batchMatMul(x:FlexTensor, y:FlexTensor):
    return ops.bmm(x.data(), y.data())

def argmax(x:FlexTensor):
    return ops.argmax(x.data(), dim=-1)

def sqrt(x):
    return x ** -0.5

def cumsum(x:Tensor, dim:int):
    return ops.cumsum(x, axis=dim)

def embedding(idx:Tensor, weight:Tensor, paddingIdx:int=0):
    return ops.embedding(idx, weight, paddingIdx)
def mha_prefill(q:Tensor, k:Tensor, v:Tensor, attentionMask:Tensor, numHead:int):
    b, s, h = q.shape
    
    assert h % numHead == 0
    headDim = h // numHead 

    scaling = headDim ** -0.5
    q = q * scaling

    global cnt
    cnt+=1

    # (b, s, nh, h1)
    q = q.view(b, s, numHead, headDim) 
    k = k.view(b, s, numHead, headDim)
    v = v.view(b, s, numHead, headDim)


    q = q.permute(0, 2, 1, 3).reshape(b*numHead, s, headDim)
    k = k.permute(0, 2, 3, 1).reshape(b*numHead, headDim, s)
    v = v.permute(0, 2, 1, 3).reshape(b*numHead, s, headDim)

    # QKT
    # output shape (b*nh, s, s)
    score = torch.bmm(q, k)

    # mask
    assert attentionMask.shape == (b, s) 
    ids = torch.arange(0, s)
    casualMask = ids <= ids.view(s, 1)
    mask = casualMask.view(1, 1, s, s) & attentionMask.view(b, 1, 1, s)

    score = score.view(b, numHead, s, s)
    score = torch.where(mask, score.view(b, numHead, s, s), -1e4) 
    score = score.view(b*numHead, s, s)
    score = F.softmax(score, dim=-1)

    # (b*nh, s, s) * (b*nh, s, h1) -> (b*nh, s, h1)
    attnOut = torch.bmm(score, v)        
    
    # (b, nh, s, h1) -> (b, s, nh, h1) -> (b, s, h)
    attnOut = attnOut.view(b, numHead, s, headDim).permute(0, 2, 1, 3).flatten(start_dim=2)

    return attnOut 
    
def mha_decode(q:Tensor, k:Tensor, v:Tensor, attentionMask:Tensor, numHead:int) :
    """
    mask : (b, s)"""
    assert q.shape[1] == 1
    b, s, h = k.shape
    # s include the token generated in this iteration

    assert h % numHead == 0
    headDim = h // numHead

    scaling = headDim ** -0.5

    # (b, 1, nh, h1)
    q = q.view(b, 1, numHead, headDim) * scaling
    k = k.view(b, s, numHead, headDim)
    v = v.view(b, s, numHead, headDim)

    # (b, 1, nh, h1) -> (b, nh, 1, h1)/(b, nh, h1, s) -> flatten dim0,1 
    q = q.permute(0, 2, 1, 3).reshape(b*numHead, 1, headDim)
    k = k.permute(0, 2, 3, 1).reshape(b*numHead, headDim, s)
    v = v.permute(0, 2, 1, 3).reshape(b*numHead, s, headDim)

    # output shape (b*nh, 1, s)
    score = torch.bmm(q, k)

    # mask, not causal mask
    assert attentionMask.shape == (b, s)
    
    score = score.view(b, numHead ,1, s)
    score = torch.where(attentionMask.view(b, 1, 1, s), score, -1e4)
    score = score.view(b*numHead, 1, s)
    score = torch.softmax(score, dim=-1)

    # (b*nh, 1, s) * (b*nh, s, h1) -> (b*nh, 1, h1)
    attnOut = torch.bmm(score, v)        
    
    # (b*nh, 1, h1) -> (b, 1, nh, h1) -> (b, 1, h)
    attnOut = attnOut.view(b, numHead, 1, headDim).permute(0, 2, 1, 3).flatten(start_dim=2)
    
    return attnOut
