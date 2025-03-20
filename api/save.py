import mindspore as ms
import numpy as np

a = ms.Tensor(input_data=[1, 2, 4])
# ms.save_checkpoint({ "a" : a}, "tmp")


b = np.zeros(shape=(3, 2))
c = np.ones(shape=(3, 2))
np.save("tmp", [b, c])

l = np.load("tmp.npy")
print(l[0])
print(l[1])