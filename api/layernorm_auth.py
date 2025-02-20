import mindspore
import numpy as np
from mindspore import Tensor, ops

input_x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]), mindspore.float32)
normalized_shape = (3,)
gamma = Tensor(np.ones(normalized_shape), mindspore.float32)
beta = Tensor(np.zeros(normalized_shape), mindspore.float32)
eps = 1e-7
output = ops.layer_norm(input_x, normalized_shape, gamma, beta, eps)
print(output)


