import numpy as np
import mindspore as ms

a = np.load( "compressedtensor.npy")
print(a)
b = ms.Tensor(a)
print(b)