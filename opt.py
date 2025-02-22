from transformers import AutoTokenizer
from mindspore import nn, ops, dtype, Parameter, common, Tensor
from mindspore.common.initializer import initializer, Zero
from mindspore import load_checkpoint
import math
import abc
import numpy as np
import os

from mindspore.numpy import ones

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from tqdm import tqdm

USING_DISK=True

class Config:
    def __init__(self, name,
            maxSeqLen, numHiddenLayer, nHead,
            hiddenSize, inputDim, ffnEmbedDim,
        ):
        self.modelName = name
        self.dtype = dtype.float32
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

def getOptConfig(name)->Config:
    if name == "opt-125m":
        config = Config(name=name,
            maxSeqLen=2048, numHiddenLayer=12, nHead=12,
            hiddenSize=768, inputDim=768, ffnEmbedDim=768 * 4,
        )
    elif name == "opt-1.3b":
        config = Config(name=name,
            maxSeqLen=2048, numHiddenLayer=24, nHead=32,
            hiddenSize=2048, inputDim=2048, ffnEmbedDim=2048 * 4,
        )
    else :
        raise NotImplementedError(f"unsupported name: {name}")

    return config


def mha_prefill(q:Tensor, k:Tensor, v:Tensor, mask:Tensor, numHead:int):
    assert q.dtype == dtype.float32
    assert k.dtype == dtype.float32
    assert v.dtype == dtype.float32

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
   

    assert attnOut.dtype == dtype.float32
    return attnOut 
    
def mha_decode(q:Tensor, k:Tensor, v:Tensor, mask:Tensor, numHead:int) :
    assert q.dtype == dtype.float32
    assert k.dtype == dtype.float32
    assert v.dtype == dtype.float32

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


class FlexTensor:
    weightHome = "weightHome"
    def __init__(self, name, shape):
        self.shape = shape
        self.name = name
        self.t = None
        self.filename = None
         
    def store(self, data:Tensor):
        assert data.dtype == dtype.float32, f"invalid dtype: {data.dtype}"

        if USING_DISK :
            self.diskStore(data)
        else :
            self.t = data

    def data(self):
        if USING_DISK:
            return self.diskData()
        else:
            assert self.t is not None
            return self.t
    def diskStore(self, data:Tensor):
        
        self.filename = os.path.join(FlexTensor.weightHome, self.name) + ".npy"
        if not os.path.exists(FlexTensor.weightHome):
            os.mkdir(FlexTensor.weightHome)

        np.save(self.filename, data.asnumpy())
    
    def diskData(self):
        assert self.filename is not None, f"disk-tensor fetch before store"
        npT = np.load(self.filename)
        return Tensor(npT)

    

class Linear:
    def __init__(self, name, inputChannel:int, outputChannel:int):
        self.name = name
        self.weight = FlexTensor(name+".weight", (outputChannel, inputChannel))
        self.bias = FlexTensor(name+".bias", (outputChannel))

    def __call__(self, x:Tensor):
        return ops.dense(x, self.weight.data(), self.bias.data())

    def getParameters(self):
        return { self.weight, self.bias }

class Layernorm:
    def __init__(self, name, normDim:int):
        self.name = name
        self.weight = FlexTensor(name+".weight", (normDim, ))
        self.bias = FlexTensor(name+".bias", (normDim, ))
        self.normDim = normDim

    def __call__(self, x:Tensor):
        l = nn.LayerNorm(normalized_shape=(self.normDim, ), 
                         gamma_init=self.weight.data(), 
                         beta_init=self.bias.data())
        
        return l(x)
    
    def getParameters(self):
        return { self.weight, self.bias }

class Attention(nn.Cell):
    def __init__(self, name, config: Config):
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

        self.layernorm = Layernorm(name+".layernorm", normDim=hiddenSize)

        self.batchMatMul = ops.BatchMatMul()    # transpose handled in construct:premute
        self.softmax = nn.Softmax()
        self.batchMatMulSV = ops.BatchMatMul()
        
        self.residual = ops.Add()

    def getParameters(self):
        return {
            *self.kProj.getParameters(),
            *self.vProj.getParameters(),
            *self.qProj.getParameters(),
            *self.outProj.getParameters(),
            *self.layernorm.getParameters()
        }

    def prefill(self, x, attentionMask):
        """ all in form (b, s, h)  """
        assert x.dtype == dtype.float32
        b, s, h = x.shape

        normalX = self.layernorm(x)
        # (b, s, h)
        q, k, v = self.qProj(normalX), self.kProj(normalX), self.vProj(normalX)
        assert q.dtype==dtype.float32

        self.kCache = k
        self.vCache = v

        # construct mask
        ids = ops.arange(0, s)
        casualMask = ids <= ids.view(s, 1)
        if attentionMask is not None:
            mask = ops.logical_and(casualMask.view(1, s, s), attentionMask.view(b, 1, s))
        else :
            mask = casualMask.view(1, s, s) 

        mhaOut = mha_prefill(q, k, v, mask, self.numHead) 

        assert mhaOut.dtype == dtype.float32
        attnOut = self.outProj(mhaOut)

        attnOut = self.residual(attnOut, x)

        return attnOut
        
    
    def decode(self, x, attentionMask):
        """
        x.shape (b, 1, h)
        attentionMask in shape (b, 1, s)
        """
        b, s, h = x.shape
        assert s == 1

        normalX = self.layernorm(x)
        # (b, s, h)
        q, k, v = self.qProj(normalX), self.kProj(normalX), self.vProj(normalX)
        self.kCache = ops.concat((self.kCache, k), axis=1)
        self.vCache = ops.concat((self.vCache, v), axis=1)
        k = self.kCache
        v = self.vCache

        mhaOut = mha_decode(q, k, v, attentionMask, self.numHead)

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
    def __init__(self, name:str, config:Config):
        super().__init__()
        self.name = name
        hiddenSize = config.hiddenSize
        ffnHiddenSize = config.ffnHiddenSize
        
        self.layernorm = Layernorm(name+".layernorm", normDim=hiddenSize)
        self.linear1 = Linear(name+".linear1", hiddenSize, ffnHiddenSize)
        self.relu = nn.ReLU()
        self.linear2 = Linear(name+".linear2", ffnHiddenSize, hiddenSize)
        self.residual = ops.Add()

    def getParameters(self):
        return {
            *self.layernorm.getParameters(),
            *self.linear1.getParameters(),
            *self.linear2.getParameters(),
        }
            

    def construct(self, x):
        assert x.dtype == dtype.float32
        o = self.layernorm(x)
        o = self.linear1(o)
        o = self.relu(o)
        o = self.linear2(o)
        
        ffnOut = self.residual(o, x)
        return ffnOut
        

class TransformerLayer(nn.Cell):
    #transformerlayer
    
    def __init__(self, name, config:Config):
        super().__init__()
        self.name = name
        self.attn = Attention(name=name+".attn", config=config)
        self.ffn = FeedForward(name=name+".ffn", config=config)

    def getParameters(self):
        return self.attn.getParameters().union(self.ffn.getParameters())

    def construct(self, x:Tensor, iterNo, attentionMask):
        assert x.dtype == dtype.float32  
        attnOut = self.attn(x, iterNo, attentionMask)
        
        ffnOut = self.ffn(attnOut) 

        assert ffnOut.dtype==dtype.float32
        return ffnOut

def lazyParameter(shape, name):
    return Parameter(
            initializer(init="normal", shape = shape),
            name = name,
        )

class InputEmbed(nn.Cell):
    #inputembed
    def __init__(self, config:Config):
        super().__init__()
        self.tokenEmbedWeight = FlexTensor(shape=(config.vocabSize, config.hiddenSize), name="inputEmbed.tokenWeight")
        self.posEmbedWeight = FlexTensor(shape=(config.maxSeqLen + 2, config.hiddenSize), name="inputEmbed.posWeight")
        self.gather = ops.operations.Gather()
        self.add = ops.Add()

    def getParameters(self):
        return { self.tokenEmbedWeight, self.posEmbedWeight }

    def construct(self, inputIDs:Tensor, attentionMask:Tensor):
        """
        inputIDs : (B, S) / (B, 1)
        each element is an idx
        attentionMask: (B, S)
        """
        assert len(inputIDs.shape) == 2
        assert isinstance(inputIDs, Tensor)
        assert isinstance(attentionMask, Tensor)
        # assert self.tokenEmbedWeight.dtype == dtype.float32
        # assert self.posEmbedWeight.dtype == dtype.float32

        tokenEmbed = self.gather(self.tokenEmbedWeight.data(), inputIDs, 0)

        posIds = ops.cumsum(attentionMask.to(dtype=dtype.int32), axis=1)
        
        posEmbed = self.gather(self.posEmbedWeight.data(), posIds, 0)

        currLength = attentionMask.shape[1]
        inputIDLength = inputIDs.shape[1]
        previousIDsLength = currLength - inputIDLength
        posEmbed = posEmbed[:, previousIDsLength:]

        assert tokenEmbed.shape == posEmbed.shape
        embed = self.add(tokenEmbed, posEmbed)
        
        assert len(embed.shape) == 3
        embed = embed.to(dtype=dtype.float32)
        assert embed.dtype == dtype.float32
        return embed

class OutputEmbed(nn.Cell):
    #outputembed
    def __init__(self, config:Config):
        super().__init__()
        self.tokenWeight = FlexTensor(shape=(config.vocabSize, config.hiddenSize), name="outputEmbed.tokenWeight")
        
        # self.norm = nn.LayerNorm(normalized_shape=(config.hiddenSize, ), 
        #                          gamma_init=lazyParameter(shape=(config.hiddenSize, ), name="output_embed_layer_norm.weight"),
        #                          beta_init=lazyParameter(shape=(config.hiddenSize), name="output_embed_layer_norm.bias"), 
        #                          dtype=dtype.float32)
        self.layernorm = Layernorm("outputEmbed.layernorm", config.hiddenSize)    
        self.matmul = ops.BatchMatMul(transpose_b=True)
        self.argmax = ops.Argmax()

    def getParameters(self):
        return self.layernorm.getParameters().union({self.tokenWeight})
    
    def construct(self, x):
        # assert x.dtype == dtype.float32
        assert self.tokenWeight.data().dtype == dtype.float32, f"invalid dtype: {self.tokenWeight.dtype}"

        normalized = self.layernorm(x)
        
        output = self.matmul(normalized, self.tokenWeight.data())
        # print(f">>> before argmax, output[0] is {output[0]}, shape: {output[0].shape}")  # (vocab)
        
        outputIDs = self.argmax(output)
        # print(f">>> after argmax, output[0] is {outputIDs[0]}, shape: {outputIDs[0].shape}")
        
        assert len(outputIDs.shape) == 2   # output shape: (B, S), element is id
        return outputIDs
    
    
class OPT(nn.Cell):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.numLayers = config.numHiddenLayer  
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, padding_side="left") 
        
        layers = nn.SequentialCell()
        self.inputEmbed = InputEmbed(config)

        for i in range(config.numHiddenLayer):
            layers.append(
                TransformerLayer(f"layers.{i}", config = config)
            )
        self.outputEmbed = OutputEmbed(config)  
        
        self.layers = layers
        
        self.loadWeight(config.weightFname)

        self.tokensBuffer: Tensor = None
        self.maxSeqLen = config.maxSeqLen
        self.attentionMask :Tensor = None   # true : valid, false : neglect

    def getParameters(self):
        ret = self.inputEmbed.getParameters()
        for l in self.layers:
            ret = ret.union(l.getParameters())
        ret = ret.union(self.outputEmbed.getParameters())
        return ret
        

    def loadWeight(self, weightFname):
        assert isinstance(weightFname, str)
        
        print(">>> load weight begin")
        weights = load_checkpoint(weightFname)
        
        uninitializedInNet = []
        unusedWeight = set(weights.keys())

        for p in self.getParameters():
            if p.name in weights.keys():
                p.store(weights[p.name].to(dtype.float32))
                unusedWeight.remove(p.name)

            else :
                uninitializedInNet.append(p.name)
                
        print("<<< load weight finish")
            
        if uninitializedInNet:
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
        for l in self.layers:
            h = l(h, i, self.attentionMask) 
        # o should be in shape (b, )
        
        h = h[:, -1:]
        o = self.outputEmbed(h)
        
        self.tokensBuffer = ops.concat((self.tokensBuffer, o), axis=1)
        self.attentionMask = ops.concat(
            (self.attentionMask, Tensor(np.ones(shape=(B,  1), dtype=np.bool_)) ), 
            axis=1)
            
    def run(self, inputSentences: list[str]):
        promptLen = max([len(l) for l in inputSentences])
        inputTokens = self.tokenizer(inputSentences, padding="max_length",  max_length=promptLen).input_ids
        self.tokensBuffer = Tensor(inputTokens, dtype=dtype.int32) 

        maxIter = 16 
        
        # init attention mask
        self.attentionMask = (self.tokensBuffer != self.config.padTokenID)
        assert isinstance(self.attentionMask, Tensor)
        
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

    model = OPT(config)

    # inputs = [
    #     "The largest cat in the world is",
    #     "The largest cat in the world is"
    # ]
    inputs = [
        "Pairs is the capital city of",
        "Pairs is the capital city of",
    ]

    outputs = model.run(inputs)
    for s in outputs:
        print(s)
