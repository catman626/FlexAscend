import mindspore as ms
import numpy as np



t = ms.Tensor(np.zeros((6,2,3)))
a = ms.Parameter(ms.common.initializer.initializer(init=t, shape=(6,2,3)))
g = ms.ops.operations.Gather()
idx = ms.Tensor([1, 3, 5])
b = g(a, idx, 0)
print(a)
print(b)