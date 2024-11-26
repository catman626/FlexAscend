from mindspore import Tensor
import numpy as np


a = np.arange(12)
a = Tensor(a)
a = a.reshape(3, 4)
print(a)
a = a.view(3 ,2, 2)
print(a)