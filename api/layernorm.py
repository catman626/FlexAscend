from mindspore import Tensor, dtype
from mindspore import ops
import numpy as np

a = Tensor(np.random.random(10), dtype=dtype.float16)
w = Tensor((0,)*10, dtype=dtype.float16)
b = Tensor((1, )*10, dtype=dtype.float16)
print(a, b, w)

print(type(a), type(b), type(w))
# layernorm = ops.LayerNorm(0, 0)
# c = layernorm(a, w, b)

c = ops.layer_norm(a, (10,), w, b)
print(c)