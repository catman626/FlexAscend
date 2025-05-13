from mindspore import Tensor, dtype
from mindspore.ops import min as msMin
from mindspore.ops import max as msMax
from mindspore.ops import clamp, mul,  round, bitwise_or, stack, div
uint8 = dtype.uint8

def minTensor(t, dim, keepdim=False):
    return msMin(t, axis=dim, keepdims=keepdim)

def maxTensor(t, dim, keepdim=False):
    return msMax(t, axis=dim, keepdims=keepdim)

import numpy as np

class compressConfig:
    nbit=4
    dim = -1
    groupSize = 32

def printAvg(a, name):
    print(f" >>> {name} average: {a.mean()}, dtype: {a.dtype}")

def lastDimEvenSlice(shape):
    """
    """
    prefixShape = shape[:-1]
    lastDim = shape[-1]
    prefixSlice = tuple(slice(0, r) for r in prefixShape)
    upperSlice = prefixSlice + (slice(0, lastDim, 2), )
    lowerSlice = prefixSlice + (slice(1, lastDim, 2), )
    return upperSlice, lowerSlice

def compress(data:Tensor):
    """
    all compress done based on Tensor
    to get Ascend support
    """

    # b, s, h = data.shape
    # assert h % compressConfig.groupSize == 0, f"invalid hidden dim: {h}, not divisable by groupSize: {compressConfig.groupSize}"
    assert isinstance(data, Tensor)

    prefixShape = data.shape[:-1]
    h = data.shape[-1]
    nGroup = h // compressConfig.groupSize
    newShape = prefixShape + (nGroup, compressConfig.groupSize)

    # print(f"new shape is: {newShape}")
    data = data.reshape(newShape)  # eg. (b, s, ng, g) ->  (b, s, h)

    mx = maxTensor(data, dim=-1, keepdim=True)[0]
    mn = minTensor(data, dim=-1, keepdim=True)[0]
    
    r = 2 ** compressConfig.nbit - 1
    scale = r / (mx-mn+1e-9)

    compressed = (data - mn) 
    compressed = mul(compressed, scale)
    compressed = clamp(compressed, min=0, max=r)
    compressed = round(compressed) #.to(dtype=dtype.uint8)
    compressed = compressed.to(uint8)

    upperSlice, lowerSlice = lastDimEvenSlice(data.shape)
    upper :Tensor = compressed[upperSlice].bitwise_left_shift(4)
    lower :Tensor = compressed[lowerSlice]
    compressed = bitwise_or(upper, lower)

    extra = stack([scale, mn])
    
    return compressed, extra
    
def decompress(data, extra, shape):
    """
    input is all on Ascend
    data / extra is ms.tensor
    """
    ng, halfg = data.shape[-2],  data.shape[-1]
    g = 2*halfg
    originShape = data.shape[:-2] + (ng, g) # shape before bitwise compression
    assert g == compressConfig.groupSize
    assert g * ng == shape[-1]

    scale, mn = extra[0], extra[1]
    upper = data.bitwise_right_shift(4)
    lower = data.bitwise_and(15)
    data = Tensor(np.zeros(dtype=np.uint8, shape=originShape))

    upperSlice, lowerSlice = lastDimEvenSlice(data.shape)
    data[upperSlice] = upper
    data[lowerSlice] = lower

    data = div(data, scale)
    data = mn + data

    originShape = data.shape[:-2] + (ng*g, )

    return data.view(originShape)



def testCompressWithShape(shape):
    # a = np.random.rand(32, 20, 256).astype(np.float32)
    a = np.random.rand(*shape).astype(np.float32)
    a = Tensor(a)
    c, extra = compress(a)
    a1 = decompress(c, extra, a.shape)
    diff = a - a1

    printAvg(diff, "diff")
    print(f"input size: {toNumpy(a).nbytes} bytes")
    print(f"output size: {toNumpy(c).nbytes} bytes")


def testCompress():
    testCompressWithShape((32, 20, 256))
    testCompressWithShape((768, 768))

if __name__ == "__main__":
    testCompress()