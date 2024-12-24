from mindspore import Tensor
import numpy as np


a = np.arange(12)
a = Tensor(a)
a = a.reshape(3, 4)
print(a)
a = a.view(3 ,2, 2)
print(a)

mask = Tensor(np.ones((2, 16)))
print(f"check if unsqueeze works: {mask.view(2, 1, 1, 16).shape}")
print(f"check if broadcast works: {mask.view(2, 1, 16, 16).shape}")