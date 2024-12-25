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

class Config:
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
        self.numHeads= nHead
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


class Attention(nn.Cell):
    def __init__(self, config: Config):
        super().__init__()

        self.headDim = headDim = config.hiddenSize // config.numHeads
        self.seqLength = config.maxSeqLen
        hiddenSize = config.hiddenSize
        self.normFactor = math.sqrt(self.headDim)
        self.numHeads = config.numHeads
        
        self.qProj = nn.Dense(hiddenSize, hiddenSize)
        self.kProj = nn.Dense(hiddenSize, hiddenSize)
        self.vProj = nn.Dense(hiddenSize, hiddenSize)

        self.outProj = nn.Dense(hiddenSize, hiddenSize)

        self.attnLayerNorm = nn.LayerNorm(normalized_shape=(hiddenSize, ))

        self.batchMatMul = ops.BatchMatMul()    # transpose handled in construct:premute
        self.softmax = nn.Softmax()
        self.batchMatMulSV = ops.BatchMatMul()
        
        self.residual = ops.Add()

    def prefill(self, x, attentionMask=None):
        """ all in form (b, s, h)  """
        b, s, h = x.shape

        normalX = self.attnLayerNorm(x)
        # (b, s, h)
        q, k, v = self.qProj(normalX), self.kProj(normalX), self.vProj(normalX)

        self.kCache = k
        self.vCache = v

        # (b, s, nh, h1)
        q = q.view(b, s, self.numHeads, self.headDim)
        k = k.view(b, s, self.numHeads, self.headDim)
        v = v.view(b, s, self.numHeads, self.headDim)

        q = ops.permute(q, (0, 2, 1, 3))
        k = ops.permute(k, (0, 2, 3, 1))
        v = ops.permute(v, (0, 2, 1, 3))

        # QKT
        # output shape (b, nh, s, s)
        score = self.batchMatMul(q, k)

        # mask
        ids = ops.arange(0, s)
        casualMask = ids <= ids.view(s, 1)
        if attentionMask is not None:
            mask = casualMask.view(1, 1, s, s) & attentionMask.view(b, 1, s, s)
        else :
            mask = casualMask.view(1, 1, s, s) 
        
        score = ops.where(mask, score, -1e4)
        score = self.softmax(score)

        # (b, nh, s, s) * (b, nh, s, h1) -> (b, nh, s, h1)
        attnOut = self.batchMatMulSV(score, v)        
        
        # (b, nh, s, h1) -> (b, s, nh, h1) -> (b, s, h)
        attnOut = ops.permute(attnOut, (0, 2, 1, 3)).flatten(start_dim=2)
        
        attnOut = self.outProj(attnOut)

        attnOut = self.residual(attnOut, x)

        return attnOut
        
    
    def decode(self, x, iterNo, attentionMask=None):
        """
        x.shape (b, 1, h)
        attentionMask in shape (b, 1, s)
        """
        b, s, h = x.shape
        assert s == 1

        normalX = self.attnLayerNorm(x)
        # (b, s, h)
        q, k, v = self.qProj(normalX), self.kProj(normalX), self.vProj(normalX)

        # q,k,v in shape (b, 1, h)
        assert k.shape[1] == 1, f"k.shape is {k.shape}"
        self.kCache = ops.concat((self.kCache, k), axis=1)
        self.vCache = ops.concat((self.vCache, v), axis=1)
        k = self.kCache
        v = self.vCache
        s = self.kCache.shape[1]
        
        # s include the token generated in this iteration

        # (b, 1, nh, h1)
        q = q.view(b, 1, self.numHeads, self.headDim)
        k = k.view(b, s, self.numHeads, self.headDim)
        v = v.view(b, s, self.numHeads, self.headDim)

        q = ops.permute(q, (0, 2, 1, 3))
        k = ops.permute(k, (0, 2, 3, 1))
        v = ops.permute(v, (0, 2, 1, 3))

        # output shape (b, nh, s, s)
        score = self.batchMatMul(q, k)

        # mask
        if attentionMask is not None:
            score = ops.where(attentionMask.view(b, 1, 1, s), score, -1e4)
        score = self.softmax(score)

        # (b, nh, 1, s) * (b, nh, s, h1) -> (b, nh, 1, h1)
        attnOut = self.batchMatMulSV(score, v)        
        
        # (b, nh, 1, h1) -> (b, 1, nh, h1) -> (b, 1, h)
        attnOut = ops.permute(attnOut, (0, 2, 1, 3)).flatten(start_dim=2)

        attnOut = self.outProj(attnOut)
        attnOut = self.residual(attnOut, x)

        return attnOut

    def construct(self, x, iterNo, attentionMask=None):
        assert attentionMask is None or len( attentionMask.shape ) == 2
        if iterNo == 0:
            return self.prefill(x, attentionMask)
        else :
            return self.decode(x, iterNo, attentionMask)
        

class FeedForward(nn.Cell):
    def __init__(self, config:Config):
        super().__init__()
        hiddenSize = config.hiddenSize
        ffnHiddenSize = config.ffnHiddenSize
        
        self.layerNorm = nn.LayerNorm(normalized_shape=(hiddenSize, ))
        self.linear1 = nn.Dense(hiddenSize, ffnHiddenSize)
        self.relu = nn.ReLU()
        self.linear2 = nn.Dense(ffnHiddenSize, hiddenSize)
        self.residual = ops.Add()

    def construct(self, x):
        o = self.layerNorm(x)
        o = self.linear1(o)
        o = self.relu(o)
        o = self.linear2(o)
        
        ffnOut = self.residual(o, x)
        return ffnOut
        

class TransformerLayer(nn.Cell):
    
    def __init__(self, config:Config):
        super().__init__()
        self.attn = Attention(config=config)
        self.ffn = FeedForward(config=config)

    def construct(self, x, iterNo):
        attnOut = self.attn(x, iterNo)
        
        ffnOut = self.ffn(attnOut) 

        return ffnOut

def lazyParameter(shape, name):
    return Parameter(
            initializer(init="normal", shape = shape),
            name = name
        )

class InputEmbed(nn.Cell):
    def __init__(self, config:Config):
        super().__init__()
        self.tokenEmbedWeight = lazyParameter(shape=(config.vocabSize, config.hiddenSize), name="embed_tokens.weight")
        self.posEmbedWeight = lazyParameter(shape=(config.maxSeqLen + 2, config.hiddenSize), name="embed_positions.weight")
        self.gather = ops.operations.Gather()
        self.add = ops.Add()

    def construct(self, inputIDs:Tensor, attentionMask:Tensor):
        """
        inputIDs : (B, S) / (B, 1)
        each element is an idx
        attentionMask: (B, S)
        """
        assert len(inputIDs.shape) == 2
        assert isinstance(inputIDs, Tensor)
        assert isinstance(attentionMask, Tensor)
        
        tokenEmbed = self.gather(self.tokenEmbedWeight, inputIDs, 0)

        posIds = ops.cumsum(attentionMask, axis=1)
        
        posEmbed = self.gather(self.posEmbedWeight, posIds, 0)

        currLength = attentionMask.shape[1]
        inputIDLength = inputIDs.shape[1]
        previousIDsLength = currLength - inputIDLength
        posEmbed = posEmbed[:, previousIDsLength:]

        assert tokenEmbed.shape == posEmbed.shape
        embed = self.add(tokenEmbed, posEmbed)
        
        assert len(embed.shape) == 3
        return embed

class OutputEmbed(nn.Cell):
    def __init__(self, config:Config):
        super().__init__()
        self.tokenWeight = lazyParameter(shape=(config.vocabSize, config.hiddenSize), name="embed_tokens.weight.ref")
        self.norm = nn.LayerNorm(normalized_shape=(config.hiddenSize, ), 
                                 gamma_init=lazyParameter(shape=(config.hiddenSize, ), name="output_embed_layer_norm.weight"),
                                 beta_init=lazyParameter(shape=(config.hiddenSize), name="output_embed_layer_norm.bias")
)
        self.matmul = ops.BatchMatMul(transpose_b=True)
        self.argmax = ops.Argmax()
    
    def construct(self, x):
        normalized = self.norm(x)
        
        output = self.matmul(normalized, self.tokenWeight)
        print(f">>> before argmax, output[0] is {output[0]}, shape: {output[0].shape}")  # (vocab)
        
        outputIDs = self.argmax(output)
        print(f">>> after argmax, output[0] is {outputIDs[0]}, shape: {outputIDs[0].shape}")
        
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
                TransformerLayer(config = config)
            )
        self.outputEmbed = OutputEmbed(config)  
        
        self.layers = layers
        
        self.loadWeight(config.weightFname)

        self.tokensBuffer: Tensor = None
        self.maxSeqLen = config.maxSeqLen
        self.attentionMask :Tensor = None   # true : valid, false : neglect
        

    def loadWeight(self, weightFname):
        assert isinstance(weightFname, str)
        
        print(">>> load weight begin")
        weights = load_checkpoint(weightFname)
        
        uninitializedInNet = []
        unusedWeight = set(weights.keys())
        for name, param in tqdm(self.parameters_and_names()):
            if name not in weights.keys():
                uninitializedInNet.append(name)
            else:
                param.set_data(weights[name])
                unusedWeight.remove(name)
                
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
        
        print(f">>> input mask: {self.attentionMask}")
        h = self.inputEmbed(inputIDs, self.attentionMask)
        for l in self.layers:
            h = l(h, i) 
        # o should be in shape (b, )
        h = h[:, -1:]
        o = self.outputEmbed(h)
        print(f">>> o inshape {o.shape}")
        
        self.tokensBuffer = ops.concat((self.tokensBuffer, o), axis=1)
        self.attentionMask = ops.concat(
            (self.attentionMask, Tensor(np.ones(shape=(B,  1), dtype=np.int32)) ), 
            axis=1)
            
    def run(self, inputSentences: list[str]):
        promptLen = max([len(l) for l in inputSentences])
        inputTokens = self.tokenizer(inputSentences, padding="max_length",  max_length=promptLen).input_ids
        self.tokensBuffer = Tensor(inputTokens, dtype=dtype.int32) 
        print(f"input tokens is: {inputTokens}")

        maxIter = 16 
        
        # init attention mask
        self.attentionMask = (self.tokensBuffer != self.config.padTokenID).to(dtype.int32)
        assert isinstance(self.attentionMask, Tensor)
        
        print(">>> inference begin")
        print(f">>> tokensBuffer inshape {self.tokensBuffer.shape}")
        for i in range(maxIter):
            print(f"    >>> loop {i} begin")
            print(f"    >>> tokensBuffer inshape {self.tokensBuffer.shape}")
            self.runIter(i, promptLen+i)
            

        print("<<< inference end")
        print(f">>> final tokens length: {self.tokensBuffer.shape}")
        print(f">>> final tokens: {self.tokensBuffer}")
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

    inputs = [
        "the largest cat in the world is",
        "the largest cat in the world is"
    ]

    outputs = model.run(inputs)
    for s in outputs:
        print(s)
