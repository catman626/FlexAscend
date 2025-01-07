import mindspore as ms
from mindspore.hal import get_device_properties

properties = get_device_properties(0, device_target="Ascend")

print(properties)