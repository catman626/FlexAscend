import mindspore as ms
from mindspore.hal import get_device_properties
from mindspore import context

context.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)

properties = get_device_properties(0, device_target="Ascend")

print(properties)