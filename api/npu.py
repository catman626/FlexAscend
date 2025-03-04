import mindspore
from mindspore import Tensor, nn
import numpy as np

x = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
h = 1000000
net = nn.Dense(3, h)
net2 = nn.Dense(h, h)
net3 = nn.Dense(h, 3)

x = net(x)
x = net2(x)
x = net3(x)
print(x.shape)

