import mindspore as ms
import numpy as np
x = ms.Tensor(np.ones([20, 5, 10, 10]), ms.float32)
shape1 = x.shape[1:]
m = ms.nn.LayerNorm(shape1,  begin_norm_axis=1, begin_params_axis=1)
output = m(x).shape
print(output)

