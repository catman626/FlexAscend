import mindspore as ms
device_target = ms.context.get_context("device_target")
print(ms.hal.get_device_properties(0, device_target))
