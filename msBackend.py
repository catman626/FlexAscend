import mindspore as ms
from mindspore import ops

class AscendTensor:
    def __init__(self):
        self.val = None
    
    def store(self, data:Tensor):
        self.val = data
        
    def data(self):
        return self.val
        
class CPUTensor:
    def __init__(self):
        self.val = None
        
    def store(self, data:Tensor):
        self.val = data

    def data(self):
        return self.val
    
class DiskTensor:
    weightHome = "weightHome"
    compress = False
    def __init__(self, name:str):
        self.name = name
        self.filename = None
        self.cached = None
        
    def store(self, data:Tensor):
        assert isinstance(data, Tensor)
        self.filename = os.path.join(DiskTensor.weightHome, self.name) + ".npy"
        if not os.path.exists(DiskTensor.weightHome):
            os.mkdir(DiskTensor.weightHome)

        self.shape = data.shape

        if DiskTensor.compress:
            data, extra = compress(data)
            
            np.save(self.filename, data.asnumpy())
            np.save(self.filename+".extra", extra.asnumpy())
        else:
            if isinstance(data, Tensor):
                data = data.asnumpy()
            np.save(self.filename, data)

    def load(self):
        assert self.filename is not None, f"disk-tensor fetch before store"
        t = np.load(self.filename)
        t = Tensor(t)
        
        if DiskTensor.compress:
            extra = np.load(self.filename + ".extra.npy") 
            extra = Tensor(extra)

            self.cached = decompress(t, extra, self.shape)
        else:
            self.cached = t

    def data(self):
        if self.cached is not None: 
            cached = self.cached
            self.cached = None
            return cached 

        self.load()
        return self.data()
        
class FlexTensor:
    def __init__(self, name, shape, home:str="DISK"):
        self.shape = shape
        self.name = name
        self.home = home    # where the data is stored, Ascend or CPU, DISK
        
        if self.home == "Ascend":
            self.tensor = AscendTensor()
        elif self.home == "CPU":
            self.tensor = CPUTensor()
        elif self.home == "DISK":
            self.tensor = DiskTensor(self.name)
        else:
            raise NotImplementedError(f"not implemented home: {self.home}")

    def store(self, data:Tensor):
        assert data.dtype == dtype.float32, f"invalid dtype: {data.dtype}"
        self.tensor.store(data)

    def load(self):
        self.tensor.load()

    def data(self):
        return self.tensor.data()

    def initZeros(self):
        return self.tensor.store(Tensor(np.zeros(self.shape, dtype=np.float32)))

        


def makeMask(attentionMask, s):

    ids = ops.arange(0, s)
    casualMask = ids <= ids.view(s, 1)
    if attentionMask is not None:
        mask = logicalAnd(casualMask.view(1, s, s), attentionMask.view(b, 1, s))
    else :
        mask = casualMask.view(1, s, s) 

    return mask

#  todo 
l = layernorm(normalized_shape=(self.normDim, ), 
                    weight=self.weight, 
                    bias=self.bias)