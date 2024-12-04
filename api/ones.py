from mindspore.numpy import ones
from mindspore import Tensor, ops
from mindspore import dtype
import numpy as np

a = ones(shape=(1,2,3), dtype=dtype.int32)
print(a)
print(f"ms.np.ones output type: {type(a)}")

b = Tensor(np.ones((1, 2,3)))
print(b)
print(f"Tensor(np.ones()) output type: {type(b)}")

c = ops.ones((1, 2, 3))
print(f"Tensor(ops.ones()) output type: {type(c)}")