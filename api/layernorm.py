from mindspore import Tensor, dtype
from mindspore import ops, nn
import numpy as np

'''
the ops.layernorm does not work
demo given in the doc fails
do not try again 

Use LayerNorm(xxx)() otherwise
'''

a = Tensor(np.random.random(10), dtype=dtype.float16)
w = Tensor((0,)*10, dtype=dtype.float16)
b = Tensor((1, )*10, dtype=dtype.float16)
print(a, b, w)

print(type(a), type(b), type(w))
# layernorm = ops.LayerNorm(0, 0)
# c = layernorm(a, w, b)

c = ops.layer_norm(a, (10,), w, b)
print(c)


""" this api works"""
x = Tensor(np.ones([20, 5, 10, 10]), dtype=dtype.float32)

shape1 = x.shape[1:]
m = nn.LayerNorm(shape1,  begin_norm_axis=1, begin_params_axis=1)
output = m(x).shape
print(output)