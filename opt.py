from transformers import AutoTokenizer
from mindspore import nn, ops, dtype, Parameter, common, Tensor
from mindspore.common.initializer import initializer, Zero
from mindspore import load_checkpoint
from mindspore import context
import mindspore as ms
from timer import timers

import math
import abc
import numpy as np
import os

from mindspore.numpy import ones

os.environ["TOKENIZERS_PARALLELISM"] = "false"
USING_DISK=False
DUMMY_WEIGHT = False    
import argparse
from tqdm import tqdm

class OptConfig:
    def __init__(self, name,
            maxSeqLen, numHiddenLayer, nHead,
            hiddenSize, inputDim, ffnEmbedDim,
        ):
        self.modelName = name
        self.dtype = dtype.float16
        self.hasBias = True
        self.maxSeqLen= maxSeqLen
        self.inputLen = inputDim
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
            hiddenSize=2048, inputDim=2048, ffnEmbedDim=2048 * 4,
        )
    elif name == "opt-2.7b":
        config = OptConfig(name=name,
            maxSeqLen=2048, numHiddenLayer=32, nHead=32,
            hiddenSize=2560, inputDim=2560, ffnEmbedDim=2560 * 4,
        )
    elif name == "opt-6.7b":
        config = OptConfig(name=name,
            maxSeqLen=2048, numHiddenLayer=32, nHead=32,
            hiddenSize=4096, inputDim=4096, ffnEmbedDim=4096 * 4,
        )
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
    elif name == "opt-175b-stage":
        config = OptConfig(name=name,
            maxSeqLen=2048, numHiddenLayer=24, nHead=96,
            hiddenSize=12288, inputDim=12288, ffnEmbedDim=12288 * 4,
        )
    else :
        raise NotImplementedError(f"unsupported name: {name}")

    return config

# class FlexTensor:
#     def __init__(self, name, shape, data:Tensor=None):
#         if USING_DISK 
        
#     @abc.abstractmethod
#     def data(self):
#         pass
    
#     @abc.abstractmethod
#     def store(self, data:Tensor):
#         pass
        
# class CacheTensor(FlexTensor):
#     def __init__(self, name, shape, data:Tensor=None):

class CacheTensor:
    def __init__(self, name, shape, data:Tensor=None):
        self.name = name 
        self.shape = shape

        if data is not None:
            self.cacheData = data
     
    @classmethod
    def fromMSTensor(cls, data:Tensor, name):
        return cls(name, data.shape, data)
        
    def initZeros(self, shape):
        self.cacheData = Tensor(np.zeros(shape, dtype=np.float16))

    def data(self)->Tensor:
        return self.cacheData 

    def store(self, data:Tensor):
        self.cacheData = data

class DiskTensor:
    weightHome = "weight_cache_home"
    namePool:set = set()

    def __init__(self, name, shape, data:Tensor=None):
        assert shape 
        assert name not in DiskTensor.namePool

        DiskTensor.namePool.add(name)
        self.name = name
        self.shape = shape

        if USING_DISK:
            self.filename = FlexTensor.weightHome + "/" + name + ".npy"

        if data is not None:
            self.store(data)
        else :
            self.initZeros(shape)
        
    @classmethod
    def fromMSTensor(cls, data:Tensor, name):
        return cls(name, data.shape, data)
        
    def initZeros(self, shape):
        np.save(self.filename, np.zeros(shape, dtype=np.float16))

    def data(self)->Tensor:
        
        npT = np.load(self.filename)
        msT = Tensor.from_numpy(npT)
        return msT
            
    def store(self, data:Tensor):
        # assert data.shape == self.shape, f"self.shape: {self.shape}, data.shape:{data.shape}"
        if USING_DISK:
            if not os.path.exists(FlexTensor.weightHome):
                os.mkdir(FlexTensor.weightHome)
            np.save(self.filename, data.asnumpy())
        else:
            self.cacheData = data

FlexTensor = DiskTensor if USING_DISK else CacheTensor 

class Linear:
    def __init__(self, name, inChannel:int, outChannel:int):
        self.name = name
        self.weight : FlexTensor = FlexTensor(name=name+".weight", shape=(outChannel, inChannel))

        self.bias = FlexTensor(name=name+".bias", shape=(outChannel, ))
        
    def __call__(self, x):
        return ops.dense(x, self.weight.data(), self.bias.data())

    def getParameters(self) -> set:
        return {
            self.weight, self.bias
        }

        
class Layernorm:
    def __init__(self, name, normSize):
        self.name = name 
        self.normSize = normSize
        
        self.weight = FlexTensor(name=self.name+".weight", shape=(normSize, ))
        self.bias = FlexTensor(name=self.name+".bias", shape=(normSize, ))


    def __call__(self, x):
        assert x.shape[-1] == self.normSize
        self.layerNormOp = nn.LayerNorm(
            normalized_shape=(self.normSize, ), 
            gamma_init=self.weight.data(),
            beta_init=self.bias.data())
        return self.layerNormOp(x)

    def getParameters(self) -> set:
        return {
            self.weight, self.bias
        }

def mha_prefill(q:Tensor, k:Tensor, v:Tensor, mask:Tensor, numHead:int):
    b, s, h = q.shape
    
    assert h % numHead == 0
    headDim = h // numHead 

    scaling = headDim ** -0.5
    # (b, s, nh, h1)
    q = q.view(b, s, numHead, headDim) * scaling
    k = k.view(b, s, numHead, headDim)
    v = v.view(b, s, numHead, headDim)

    q = ops.permute(q, (0, 2, 1, 3))
    k = ops.permute(k, (0, 2, 3, 1))
    v = ops.permute(v, (0, 2, 1, 3))

    # QKT
    # output shape (b, nh, s, s)
    score = ops.bmm(q, k)

    # mask
    assert mask.shape == (b, s, s) 
    score = ops.where(mask.view(b, 1, s, s), score, -1e4) 
    score = ops.softmax(score)

    # (b, nh, s, s) * (b, nh, s, h1) -> (b, nh, s, h1)
    attnOut = ops.bmm(score, v)        
    
    # (b, nh, s, h1) -> (b, s, nh, h1) -> (b, s, h)
    attnOut = ops.permute(attnOut, (0, 2, 1, 3)).flatten(start_dim=2)
   
    return attnOut 
    
def mha_decode(q:Tensor, k:Tensor, v:Tensor, mask:Tensor, numHead:int) :
    assert q.shape[1] == 1
    b, s, h = k.shape
    # s include the token generated in this iteration

    assert h % numHead == 0
    headDim = h // numHead

    scaling = headDim ** -0.5

    # (b, 1, nh, h1)
    q = q.view(b, 1, numHead, headDim) * scaling
    k = k.view(b, s, numHead, headDim)
    v = v.view(b, s, numHead, headDim)

    # (b, 1, nh, h1) -> (b, nh, 1, h1)/(b, nh, h1, s)
    q = ops.permute(q, (0, 2, 1, 3))
    k = ops.permute(k, (0, 2, 3, 1))
    v = ops.permute(v, (0, 2, 1, 3))

    # output shape (b, nh, s, s)
    score = ops.bmm(q, k)

    # mask
    assert mask.shape == (b, s)
    score = ops.where(mask.view(b, 1, 1, s), score, -1e4)
    score = ops.softmax(score)

    # (b, nh, 1, s) * (b, nh, s, h1) -> (b, nh, 1, h1)
    attnOut = ops.bmm(score, v)        
    
    # (b, nh, 1, h1) -> (b, 1, nh, h1) -> (b, 1, h)
    attnOut = ops.permute(attnOut, (0, 2, 1, 3)).flatten(start_dim=2)
    
    return attnOut


class Attention(nn.Cell):
    #attention
    def __init__(self, name:str, config: OptConfig):
        super().__init__()
        self.name = name
        self.headDim = config.hiddenSize // config.numHead
        self.seqLength = config.maxSeqLen
        hiddenSize = config.hiddenSize
        self.normFactor = math.sqrt(self.headDim)
        self.numHead = config.numHead
        
        self.qProj = Linear(self.name+".qProj", hiddenSize, hiddenSize)
        self.kProj = Linear(self.name+".kProj", hiddenSize, hiddenSize)
        self.vProj = Linear(self.name+".vProj", hiddenSize, hiddenSize)
        self.outProj = Linear(self.name+".outProj", hiddenSize, hiddenSize)
        self.layernorm = Layernorm(self.name+".layernorm", hiddenSize)

        self.softmax = nn.Softmax()
        
        self.residual = ops.Add()

    def getParameters(self) -> set:
        ret = set()
        for p in [self.qProj, self.kProj, self.vProj, self.outProj, self.layernorm]:
            ret = ret.union(p.getParameters())

        return ret


    def prefill(self, x, attentionMask):
        """ all in form (b, s, h)  """
        b, s, h = x.shape


        print(f"\t\t>>> before layernorm: {x}", end="\n\n")
        
        normalX = self.layernorm(x)
        # (b, s, h)
        print(f"\t\t>>> after layernorm: {x}", end="\n\n")
        
        q = self.qProj(normalX)
        k = self.kProj(normalX)
        v = self.vProj(normalX)
        
        print(f"\t\t after projection: {q}")

        self.kCache = FlexTensor.fromMSTensor(k, self.name+".kcache")
        self.vCache = FlexTensor.fromMSTensor(v, self.name+".vcache")

        # construct mask
        ids = ops.arange(0, s)
        casualMask = ids <= ids.view(s, 1)
        if attentionMask is not None:
            mask = ops.logical_and(casualMask.view(1, s, s), attentionMask.view(b, 1, s))
        else :
            mask = casualMask.view(1, s, s) 

        # calculate prefill 
        mhaOut = mha_prefill(q, k, v, mask, self.numHead) 
        
        print(f"after mha: {mhaOut}", end="\n\n")

        attnOut = self.outProj(mhaOut)

        attnOut = self.residual(attnOut, x)

        return attnOut
        
    
    def decode(self, x, attentionMask):
        """
        x.shape (b, 1, h)
        attentionMask in shape (b, 1, s)
        """
        b, s, h = x.shape
        assert s == 1, f"invalid decode BS: {s}"

        normalX = self.layernorm(x)
        # (b, s, h)
        q = self.qProj(normalX)
        k = self.kProj(normalX)
        v = self.vProj(normalX)

        kcache = ops.concat([self.kCache.data(), k], axis=1)
        vcache = ops.concat([self.vCache.data(), v], axis=1)
        self.kCache.store(kcache)
        self.vCache.store(vcache)

        mhaOut = mha_decode(q, kcache, vcache, attentionMask, self.numHead)
        
        attnOut = self.outProj(mhaOut)

        attnOut = self.residual(attnOut, x)

        return attnOut

    def construct(self, x, iterNo, attentionMask):
        assert attentionMask is None or len( attentionMask.shape ) == 2

        if iterNo == 0:
            return self.prefill(x, attentionMask)
        else :
            return self.decode(x, attentionMask)
        

class FeedForward(nn.Cell):
    #ffn
    def __init__(self, parentName:str, config:OptConfig):
        super().__init__()
        self.name = parentName
        hiddenSize = config.hiddenSize
        ffnHiddenSize = config.ffnHiddenSize
        
        self.linear1:Linear = Linear(self.name+".linear1", hiddenSize, ffnHiddenSize)
        self.linear2:Linear = Linear(self.name+".linear2", ffnHiddenSize, hiddenSize)
        self.layernorm = Layernorm(self.name+".layernorm", hiddenSize)

        self.relu = nn.ReLU()
        self.residual = ops.Add()

    def getParameters(self)->set:
        ret = set()
        for p in [self.linear1, self.linear2, self.layernorm]:
            ret = ret.union(p.getParameters())

        return ret
            

    def construct(self, x):
        o = self.layernorm(x)
        o = self.linear1(o)
        o = self.relu(o)
        o = self.linear2(o)
        
        ffnOut = self.residual(o, x)
        return ffnOut
        

def MSZeros(shape:tuple):
    return Tensor(np.zeros(shape=shape, dtype=np.float16))

class TransformerLayer(nn.Cell):
    #transformerlayer
    def __init__(self, name:str, config:OptConfig):
        super().__init__()
        self.name = name
        self.attn = Attention(name+".attn", config=config)
        self.ffn = FeedForward(name+".ffn", config=config)

    def construct(self, x, iterNo, attentionMask):
        # print(f"\t\t>>> attention input is: {x}")
        attnOut = self.attn(x, iterNo, attentionMask)
        # print(f"\t\t>>> attention_output/ffn_input is: {attnOut}", end="\n\n")
        
        ffnOut = self.ffn(attnOut) 
        # print(f"\t\t>>> ffn_output is: {ffnOut}", end="\n\n")

        return ffnOut

    def getParameters(self)->set:
        return self.attn.getParameters().union(
            self.ffn.getParameters()
        ) 
        
        


class InputEmbed(nn.Cell):
    # input embed
    def __init__(self, name, config:OptConfig):
        super().__init__()
        self.tokenWeight = FlexTensor(
            name=name+".tokenWeight", shape=(config.vocabSize, config.hiddenSize))
        self.posWeight = FlexTensor(
            name=name+".posWeight", shape=(config.maxSeqLen+2, config.hiddenSize))
        self.gather = ops.operations.Gather()
        self.add = ops.Add()

    def getParameters(self)->set:
        return {
            self.tokenWeight,
            self.posWeight
        }

    def construct(self, inputIDs:Tensor, attentionMask:Tensor):
        """
        inputIDs : (B, S) / (B, 1)
        each element is an idx
        attentionMask: (B, S)
        """
        assert len(inputIDs.shape) == 2
        assert isinstance(inputIDs, Tensor)
        assert isinstance(attentionMask, Tensor)
        # print(f"tokenweight shape is : ", self.tokenWeight.data().shape)
        # print(f"posWeight shape is : ", self.posWeight.data().shape)
        
        tokenEmbed = self.gather(self.tokenWeight.data(), inputIDs, 0)

        posIds = ops.cumsum(attentionMask.to(dtype=dtype.int32), axis=1)
        
        posEmbed = self.gather(self.posWeight.data(), posIds, 0)

        currLength = attentionMask.shape[1]
        inputIDLength = inputIDs.shape[1]
        previousIDsLength = currLength - inputIDLength
        posEmbed = posEmbed[:, previousIDsLength:]
        
        # print(f"        >>> pos embed: {posEmbed}")
        # print(f"        >>> token embed: {tokenEmbed}")
        
        assert tokenEmbed.shape == posEmbed.shape
        embed = self.add(tokenEmbed, posEmbed)
        
        assert len(embed.shape) == 3, f"invalid input embed shape: {embed.shape}"
        return embed

class OutputEmbed(nn.Cell):
    def __init__(self, name, config:OptConfig):
        super().__init__()
        self.tokenWeight = FlexTensor(name+".tokenWeight", shape=(config.vocabSize, config.hiddenSize))
        self.layernorm = Layernorm(name=name+".layernorm", normSize=config.hiddenSize)
        self.matmul = ops.BatchMatMul(transpose_b=True)
        self.argmax = ops.Argmax()

    def getParameters(self)->set:
        return { self.tokenWeight }.union(self.layernorm.getParameters())
    
    def construct(self, x):
        normalized = self.layernorm(x)
        
        output = self.matmul(normalized, self.tokenWeight.data())
        
        outputIDs = self.argmax(output)
        
        assert len(outputIDs.shape) == 2   # output shape: (B, S), element is id
        return outputIDs
    
    
class OPT(nn.Cell):
    def __init__(self, config:OptConfig):
        super().__init__()
        self.config = config
        self.numLayers = config.numHiddenLayer  
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, padding_side="left") 
        
        layers = nn.SequentialCell()
        self.inputEmbed = InputEmbed("inputEmbed", config)

        for i in range(config.numHiddenLayer):
            layers.append(
                TransformerLayer("layers."+str(i), config = config)
            )

        self.outputEmbed = OutputEmbed("outputEmbed", config)  
        
        self.layers = layers
        
        self.loadWeight(config.weightFname)

        self.tokensBuffer: Tensor = None
        self.maxSeqLen = config.maxSeqLen
        self.attentionMask :Tensor = None   # true : valid, false : neglect
        

    def getParameters(self)->set:
        ret = self.inputEmbed.getParameters()
        ret = ret.union(self.outputEmbed.getParameters())
        for l in self.layers:
            ret = ret.union(l.getParameters())
        return ret

    def loadWeight(self, weightFname):
        if DUMMY_WEIGHT:
            print(f">>> dummy weight, weight all zero")
            return
        assert isinstance(weightFname, str)
        
        print(">>> load weight begin")
        weights = load_checkpoint(weightFname)
        
        uninitializedInNet = []
        unusedWeight = set(weights.keys())
        
        for param in tqdm(self.getParameters()):
            pname = param.name
            if pname not in weights.keys():
                uninitializedInNet.append(pname)
            else:
                param.store(weights[pname])
                unusedWeight.remove(pname)
                
        print("<<< load weight finish")
        if sorted(uninitializedInNet):
            print(">>> uninitialized weight: ")
            for name in uninitializedInNet:
                # attention mask in shape (b, s)
                print(name)

        if unusedWeight:
            print(">>> unused weight:") 
            for name in unusedWeight:
                print(name)

        if uninitializedInNet:
            exit(1)

    def runIter(self, i, currLen):
        B = self.tokensBuffer.shape[0]
        
        # inputEmbed in shape (b, s)
        inputIDs = self.tokensBuffer[:, :currLen] if i == 0 \
            else self.tokensBuffer[:, -1:]
        
        # print(f">>> input mask: {self.attentionMask}")
        h = self.inputEmbed(inputIDs, self.attentionMask)
        for no, l in enumerate(self.layers):
            print(f"        >>> layer-{no} input is {h}", end="\n\n")
            h = l(h, i, self.attentionMask) 
            # print(f"        >>> activation {h}")
        # o should be in shape (b, )
        h = h[:, -1:]
        print(f"last hidden value is: {h}")

        o = self.outputEmbed(h)
        
        self.tokensBuffer = ops.concat((self.tokensBuffer, o), axis=1)
        self.attentionMask = ops.concat(
            (self.attentionMask, Tensor(np.ones(shape=(B,  1), dtype=np.bool_)) ), 
            axis=1)
            
    def run(self, inputSentences: list[str]):
        promptLen = max([len(l) for l in inputSentences])
        inputTokens = self.tokenizer(inputSentences, padding="max_length",  max_length=promptLen).input_ids
        self.tokensBuffer = Tensor(inputTokens, dtype=dtype.int32) 
        
        # init attention mask
        self.attentionMask = (self.tokensBuffer != self.config.padTokenID)
        assert isinstance(self.attentionMask, Tensor)
        
        maxIter = 2 
        print(">>> inference begin")
        for i in range(maxIter):
            print(f"    >>> loop {i} begin")
            self.runIter(i, promptLen+i)

        print("<<< inference end")
        outputSentences = []
        
        for line in self.tokensBuffer.tolist():
            sentence = self.tokenizer.convert_ids_to_tokens(line)
            sentence = self.tokenizer.convert_tokens_to_string(sentence)

            outputSentences.append(sentence)

        return outputSentences

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="/home/ma-user/work/FlexAscend/model/opt-1.3b")
    parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()

    config = getOptConfig(args.model)
    config.weightFname = args.ckpt
    config.tokenizer = args.tokenizer

    context.set_context(device_target="CPU", mode=ms.PYNATIVE_MODE)
    model = OPT(config)

    inputs = [
        "Pairs is the capital city of",
        "Pairs is the capital city of",
    ]

    timers("model").start()
    outputs = model.run(inputs)
    for s in outputs:
        print(s)
    timers("model").stop()
    for tName in timers.timers.keys():
        print(f"timer({tName}): {timers.timers[tName].elapsed('sum')}")

