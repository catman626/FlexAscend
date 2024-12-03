from mindspore.numpy import ones
from mindspore import Tensor
from mindspore import dtype
import numpy as np

a = ones(shape=(1,2,3), dtype=dtype.int32)
print(a)
print(type(a))

b = Tensor(np.ones((1, 2,3)))
print(b)
print(type(b))
