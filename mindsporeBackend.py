from itertools import count
import numpy as np
import mindspore as ms

class DeviceType:
    CPU = auto()
    NPU = auto()
    DISK = auto()
    
    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "npu":
            return DeviceType.NPU
        elif name == "disk":
            return DeviceType.DISK
        else:
            raise ValueError(f"invalid device name: {name}")
    

   
class MindsporeDevice:
    """
    a wrapper for mindspore tensor
    """     

    nameCount = count()

    def __init__(self, shape, dtype, data, device, name=None):
        self.shape = shape
        self.dtype = dtype
        self.data = data 
        self.device = device 
        self.name = name or MindsporeDevice.nextName()


    @classmethod
    def nextName(cls):
        return f"t_{next(cls.nameCount)}"

    @classmethod
    def from_mindspore(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name=name)


    def delete(self):
        assert self.device is not None, "already deleted!"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        self.device = None
        self.data = None

    def load_from_np(self, npArray)
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                np.save(fout, npArray)

        else :
            if self.device.deviceType == DeviceType.COMPRESSED:
                raise NotImplementedError()

            else:   
                self.data.copy_(ms.Tensor.from_numpy(npArray)) 


    def copy(self, dst, srcIndices=None):
        if srcIndices:
            assert all(x.step is None for x in srcIndices)

            shape = tuple(x.stop - x.start for x in srcIndices) + self.shape[len(srcIndices):]

        else :
            shape = self.shape

        if self.device.deviceType == DeviceType.COMPRESSED:
            raise NotImplementedError()

        else :
            ret = dst.allocate(shape, ms.dtype_to_nptype[self.dtype])
        generalCopy(dst, None, self, srcIndices)
        return ret

    def __str__(self):
        return (f"MindsporeTensor(shape={self.shape}, dtype = {str(self.dtype)}, device={self.device.name if self.device else None})")

        
class MindsporeDevice:
    
    def __init__(self, name, memCapacity=None, flops=None):

        self.name = name
        self.memCapacity = memCapacity
        self.flops = flops

        self.dev = ms.

        
        

        