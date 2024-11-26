from mindspore import Tensor, ops, int32
import numpy as np


a = Tensor(np.arange(10))
b = ops.where(a <= a.view(10, 1), 1, -1e4)
print(b.to(dtype=int32))

