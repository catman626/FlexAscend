import mindspore as ms
from mindspore import Tensor, context
import numpy as np

context.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)

a = Tensor(np.zeros((1, 2, 3)))

context.set_context(device_target="CPU", mode=ms.PYNATIVE_MODE)
b = Tensor(np.zeros((1, 2, 3)))

c = a + b

print(c)
