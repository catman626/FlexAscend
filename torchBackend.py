import torch
from torch import Tensor, dtype
from torch import arange, cumsum, argmax, concat, bmm
from torch.nn import ReLU
import torch.nn.functional as F 
from torch.nn.functional import linear
import numpy as np
import os
from compress import compress


class AscendTensor:
    def __init__(self):
        self.val = None
    
    def store(self, data:Tensor):
        self.val = data
        
    def data(self):
        return self.val
        
class CPUTensor:
    def __init__(self):
        self.val = None
        
    def store(self, data:Tensor):
        self.val = data

    def data(self):
        return self.val
    
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
            
            torch.save(self.filename, data.asnumpy())
            np.save(self.filename+".extra", extra.asnumpy())
        else:
            if isinstance(data, Tensor):
                data = data.asnumpy()
            np.save(self.filename, data)

    def load(self):
        assert self.filename is not None, f"disk-tensor fetch before store"
        t = np.load(self.filename)
        t = Tensor(t)
        
        if DiskTensor.compress:
            extra = np.load(self.filename + ".extra.npy") 
            extra = Tensor(extra)

            self.cached = decompress(t, extra, self.shape)
        else:
            self.cached = t

    def data(self):
        if self.cached is not None: 
            cached = self.cached
            self.cached = None
            return cached 

        self.load()
        return self.data()
        
class FlexTensor:
    def __init__(self, name, shape, home:str="DISK"):
        self.shape = shape
        self.name = name
        self.home = home    # where the data is stored, Ascend or CPU, DISK
        
        if self.home == "Ascend":
            self.tensor = AscendTensor()
        elif self.home == "CPU":
            self.tensor = CPUTensor()
        elif self.home == "DISK":
            self.tensor = DiskTensor(self.name)
        else:
            raise NotImplementedError(f"not implemented home: {self.home}")

    def store(self, data:Tensor):
        assert data.dtype == dtype.float32, f"invalid dtype: {data.dtype}"
        self.tensor.store(data)

    def load(self):
        self.tensor.load()

    def data(self):
        return self.tensor.data()

    def initZeros(self):
        return self.tensor.store(Tensor(np.zeros(self.shape, dtype=np.float32)))

def batchMatMul(A, B, transposeA=False, transposeB=False):
    Ad = A.data()
    Bd = B.data()
    if transposeA :
        Ad = Ad.T
    if transposeB:
        Bd = Bd.T

    return torch.bmm(Ad, Bd)


def layernorm(x:FlexTensor, normalizedShape:FlexTensor, weight:FlexTensor, bias:FlexTensor):
    return F.layer_norm(x.data(), normalizedShape, weight.data(), bias.data())

def batchMatMul(x:FlexTensor, y:FlexTensor):
    return torch.bmm(x.data(), y.data())

def argmax(x:FlexTensor):
    return torch.argmax(x.data(), dim=-1)

def sqrt(x):
    return x ** -0.5

def makeMask(attentionMask, s):

    ids = arange(0, s)
    casualMask = ids <= ids.view(s, 1)
    if attentionMask is not None:
        mask = casualMask.view(1, s, s) & attentionMask.view(b, 1, s)
    else :
        mask = casualMask.view(1, s, s) 

    return mask

def prefill(x, qProj, kProj, vProj, outProj, attentionMask, numHead, kCache, vCache):

    b, s, h = x.shape

    # (b, s, h)
    q, k, v = qProj(x), kProj(x), vProj(x)

    kCache.store(k)
    vCache.store(v)

    # make a casual mask and combine it with attention mask
    mask = makeMask(attentionMask, s)

    mhaOut = mha_prefill(q, k, v, mask, numHead) 

    attnOut = outProj(mhaOut)

    attnOut = torch.add(attnOut, x)

    return attnOut

def mha_prefill(q:Tensor, k:Tensor, v:Tensor, mask:Tensor, numHead:int):
    b, s, h = q.shape
    
    assert h % numHead == 0
    headDim = h // numHead 

    scaling = headDim ** -0.5
    # (b, s, nh, h1)
    q = q.view(b, s, numHead, headDim) * scaling
    k = k.view(b, s, numHead, headDim)
    v = v.view(b, s, numHead, headDim)

    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 3, 1)
    v = v.permute(0, 2, 1, 3)

    # QKT
    # output shape (b, nh, s, s)
    score = torch.bmm(q, k)

    # mask
    assert mask.shape == (b, s, s) 
    score = torch.where(mask.view(b, 1, s, s), score, -1e4) 
    score = F.softmax(score)

    # (b, nh, s, s) * (b, nh, s, h1) -> (b, nh, s, h1)
    attnOut = torch.bmm(score, v)        
    
    # (b, nh, s, h1) -> (b, s, nh, h1) -> (b, s, h)
    attnOut = attnOut.permute(0, 2, 1, 3).flatten(start_dim=2)
   

    assert attnOut.dtype == dtype.float32
    return attnOut 
    
def mha_decode(q:Tensor, k:Tensor, v:Tensor, mask:Tensor, numHead:int) :
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

    # (b, 1, nh, h1) -> (b, nh, 1, h1)/(b, nh, h1, s)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 3, 1)
    v = v.permute(0, 2, 1, 3)

    # output shape (b, nh, s, s)
    score = torch.bmm(q, k)

    # mask
    assert mask.shape == (b, s)
    score = torch.where(mask.view(b, 1, 1, s), score, -1e4)
    score = softmax(score)

    # (b, nh, 1, s) * (b, nh, s, h1) -> (b, nh, 1, h1)
    attnOut = torch.bmm(score, v)        
    
    # (b, nh, 1, h1) -> (b, 1, nh, h1) -> (b, 1, h)
    attnOut = attnOut.permute(0, 2, 1, 3).flatten(start_dim=2)
    
    return attnOut

def gather(weight, idx, padding):
    return F.embedding(idx, weight, padding)

def decode(x, qProj, kProj, vProj, outProj, kCache, vCache, attentionMask, numHead):
    b, s, h = x.shape
    assert s == 1

    normalX = F.layer_norm(x)
    # (b, s, h)
    q, k, v = qProj(normalX), kProj(normalX), vProj(normalX)
    kcache = kCache.data()
    vcache = vCache.data()
    k = concat((kcache, k), axis=1)
    v = concat((vcache, v), axis=1)
    kCache.store(k)
    vCache.store(v)

    mhaOut = mha_decode(q, k, v, attentionMask, numHead)

    attnOut = outProj(mhaOut)

    attnOut = attnOut + x

    return attnOut