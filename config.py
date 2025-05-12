<<<<<<< HEAD
import torch
=======
from mindspore import dtype
>>>>>>> mindspore

class OptConfig:
    def __init__(self, name,
            maxSeqLen, numHiddenLayer, nHead,
            hiddenSize, inputDim, ffnEmbedDim,
        ):
        self.modelName = name
<<<<<<< HEAD
        self.dtype = torch.float32
=======
        self.dtype = dtype.float32
>>>>>>> mindspore
        self.hasBias = True
        self.maxSeqLen= maxSeqLen
        self.inputDim = inputDim
        self.batchSize = 64

        self.numHiddenLayer = numHiddenLayer
        self.vocabSize = 50272
        self.weightFname = None
        self.localTokenizer = "opt-1.3b"
        self.numHead= nHead
        self.hiddenSize=hiddenSize
        self.ffnHiddenSize = ffnEmbedDim
        self.tokenizer = None
        self.padTokenID :int = 1

def getOptConfig(name)->OptConfig:
    if name == "opt-125m":
        config = OptConfig(name=name,
            maxSeqLen=2048, numHiddenLayer=12, nHead=12,
            hiddenSize=768, inputDim=768, ffnEmbedDim=768 * 4,
        )
    elif name == "opt-1.3b":
        config = OptConfig(name=name,
            maxSeqLen=2048, numHiddenLayer=24, nHead=32,
            hiddenSize=2048, inputDim=2048, ffnEmbedDim=2048 * 4,)
<<<<<<< HEAD
    elif name == "opt-2.7b":
        config = OptConfig(name=name,
            maxSeqLen=2048, numHiddenLayer=32, nHead=32,
            hiddenSize=2560, inputDim=2560, ffnEmbedDim=2560 * 4,
        )
=======
>>>>>>> mindspore
    elif name == "opt-6.7b":
        config = OptConfig(name=name,
            maxSeqLen=2048, numHiddenLayer=32, nHead=32,
            hiddenSize=4096, inputDim=4096, ffnEmbedDim=4096 * 4,
        )
<<<<<<< HEAD
    elif name == "opt-13b":
        config = OptConfig(name=name,
            maxSeqLen=2048, numHiddenLayer=40, nHead=40,
            hiddenSize=5120, inputDim=5120, ffnEmbedDim=5120 * 4,
        )
    elif name == "opt-30b":
        config = OptConfig(name=name,
            maxSeqLen=2048, numHiddenLayer=48, nHead=56,
            hiddenSize=7168, inputDim=7168, ffnEmbedDim=7168 * 4,
        )
=======
>>>>>>> mindspore
    elif name == "opt-66b":
        config = OptConfig(name=name,
            maxSeqLen=2048, numHiddenLayer=64, nHead=72,
            hiddenSize=9216, inputDim=9216, ffnEmbedDim=9216 * 4,
        )
    elif name == "opt-175b":
        config = OptConfig(name=name,
            maxSeqLen=2048, numHiddenLayer=96, nHead=96,
            hiddenSize=12288, inputDim=12288, ffnEmbedDim=12288 * 4,
        )
    else :
        raise NotImplementedError(f"unsupported name: {name}")

    return config