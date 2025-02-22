import mindspore as ms
from mindspore import dtype


a = ms.Tensor([1, 2, 3])
print(f"before to, type: {a.dtype}")

b = a.astype(dtype.float16)
print(f"after to, type: {b.dtype}")
print(f"equal to dtype.float16: {b.dtype == dtype.float16}")
