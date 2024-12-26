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
        self.dtype = dtype.float32
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
        self.dtype = config.dtype
        self.seqLength = config.maxSeqLen
        hiddenSize = config.hiddenSize
        self.normFactor = math.sqrt(self.headDim)
        self.numHeads = config.numHeads
        
        self.qProj = nn.Dense(hiddenSize, hiddenSize, dtype=self.dtype)
        self.kProj = nn.Dense(hiddenSize, hiddenSize, dtype=self.dtype)
        self.vProj = nn.Dense(hiddenSize, hiddenSize, dtype=self.dtype)

        self.outProj = nn.Dense(hiddenSize, hiddenSize, dtype=self.dtype)

        self.attnLayerNorm = nn.LayerNorm(normalized_shape=(hiddenSize, ))

        self.mha = nn.MultiheadAttention(embed_dim=config.hiddenSize, 
                                          num_heads=config.numHeads, 
                                          dropout=0, 
                                          has_bias=True, 
                                          batch_first=True,
                                          dtype=self.dtype)
        
        self.residual = ops.Add()

    def prefill(self, x, attentionMask):
        """ all in form (b, s, h)  """
        b, s, h = x.shape

        normalX = self.attnLayerNorm(x)
        # (b, s, h)
        q, k, v = self.qProj(normalX), self.kProj(normalX), self.vProj(normalX)

        # mask
        ids = ops.arange(0, s)
        casualMask = ids <= ids.view(s, 1)
        assert casualMask.dtype == dtype.bool_
        assert attentionMask.dtype == dtype.bool_

        # output: (b, s, h)
        mhaOut , = self.mha(q, k, v, 
                              key_padding_mask=attentionMask, 
                              need_weights = False, 
                              attn_mask=casualMask)
        
        assert isinstance(mhaOut, Tensor), f"mhaOut is not Tensor, but {type(mhaOut)}"
        
        print(f">>> mhaOut value: {mhaOut}")

        print(f">>> mhaOut value: {mhaOut}")
        o = self.outProj(mhaOut)
        o = self.residual(o, x)
        return o
        
    
    def decode(self, x, attentionMask=None):
        """
        x.shape (b, 1, h)
        attentionMask in shape (b, 1, s)
        """
        b, s, h = x.shape
        assert s == 1

        normalX = self.attnLayerNorm(x)
        # (b, s, h)
        q, k, v = self.qProj(normalX), self.kProj(normalX), self.vProj(normalX)

        mhaOut = self.mha(q, k, v,
                           key_padding_mask=attentionMask,
                           need_weights=False)

        o = self.outProj(mhaOut)
        o = self.residual(o, x)

        return o

    def construct(self, x, iterNo, attentionMask):
        assert attentionMask is None or len( attentionMask.shape ) == 2
        assert x.dtype == self.dtype

        return self.prefill(x, attentionMask)
        if iterNo == 0:
            return self.prefill(x, attentionMask)
        else :
            return self.decode(x, attentionMask)
        

class FeedForward(nn.Cell):
    def __init__(self, config:Config):
        super().__init__()
        hiddenSize = config.hiddenSize
        ffnHiddenSize = config.ffnHiddenSize
        
        self.dtype = config.dtype
        self.layerNorm = nn.LayerNorm(normalized_shape=(hiddenSize, ))
        self.linear1 = nn.Dense(hiddenSize, ffnHiddenSize, dtype=self.dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Dense(ffnHiddenSize, hiddenSize, dtype=self.dtype)
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

    def construct(self, x, iterNo, attentionMask):
        attnOut = self.attn(x, iterNo, attentionMask)
        
        ffnOut = self.ffn(attnOut) 

        return ffnOut

def lazyParameter(shape, dtype):
    return Parameter(initializer(init="normal", shape = shape, dtype=dtype))

class InputEmbed(nn.Cell):
    def __init__(self, config:Config):
        super().__init__()
        self.dtype = config.dtype
        self.tokenEmbedWeight = lazyParameter(shape=(config.vocabSize, config.hiddenSize), dtype=self.dtype)
        self.posEmbedWeight = lazyParameter(shape=(config.maxSeqLen + 2, config.hiddenSize), dtype=self.dtype)
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

        posIds = ops.cumsum(attentionMask.to(dtype.int32), axis=1)
        
        posEmbed = self.gather(self.posEmbedWeight, posIds, 0)

        currLength = attentionMask.shape[1]
        inputIDLength = inputIDs.shape[1]
        previousIDsLength = currLength - inputIDLength
        posEmbed = posEmbed[:, previousIDsLength:]

        assert tokenEmbed.shape == posEmbed.shape
        embed = self.add(tokenEmbed, posEmbed)
        
        assert len(embed.shape) == 3
        # assert embed.dtype == dtype.float16, f"invalid dtype: {embed.dtype}"
        embed = embed.to(self.dtype) 
        return embed

class OutputEmbed(nn.Cell):
    def __init__(self, config:Config):
        super().__init__()
        self.dtype = config.dtype
        self.tokenWeight = lazyParameter(shape=(config.vocabSize, config.hiddenSize), dtype=self.dtype)
        self.norm = nn.LayerNorm(normalized_shape=(config.hiddenSize, ), 
                                 gamma_init=lazyParameter(shape=(config.hiddenSize, ), dtype=self.dtype),
                                 beta_init=lazyParameter(shape=(config.hiddenSize), dtype=self.dtype), dtype=self.dtype)
        self.matmul = ops.BatchMatMul(transpose_b=True)
        self.argmax = ops.Argmax()
    
    def construct(self, x):
        normalized = self.norm(x)

        # output in shape (B, S, Vocab)
        output = self.matmul(normalized, self.tokenWeight) 
        
        outputIDs = self.argmax(output)
        
        assert len(outputIDs.shape) == 2   # outputIDs shape: (B, S), element is id
        
        print(f">>> output is: {output[:, -1:]}")
        return outputIDs
    
def mhaNameToWeightName(name) :
    pass    
    
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

        # load mha weight
        eye = np.eye(self.config.hiddenSize)
        eye3 = np.concatenate([eye, eye, eye], axis=0) 
        zeros = np.zeros((self.config.hiddenSize,))
        zeros3 = np.concatenate([zeros, zeros, zeros], axis=0) 
        
        for name, param in tqdm(self.parameters_and_names()):
            if name not in uninitializedInNet:
                continue
            if name.find("mha") == -1:
                continue
            
            if name.find("in_proj_weight") != -1:
                param.set_data(Tensor(eye3, dtype=self.config.dtype)) 
                uninitializedInNet.remove(name)

            elif name.find("in_proj_bias") != -1:
                param.set_data(Tensor(zeros3, dtype=self.config.dtype))
                uninitializedInNet.remove(name)
                
            elif name.find("out_proj.weight") != -1:
                param.set_data(Tensor(eye, dtype=self.config.dtype))
                uninitializedInNet.remove(name)

            elif name.find("out_proj.bias")  != -1:
                param.set_data(Tensor(zeros, dtype=self.config.dtype))
                uninitializedInNet.remove(name)
                
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
        inputIDs = self.tokensBuffer[:, :currLen]
        
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
        print(f"input tokens is: {inputTokens}")

        maxIter = 1 
        
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

    inputs = [
        "the largest cat in the world is",
        "the largest cat in the world is"
    ]

    outputs = model.run(inputs)
    for s in outputs:
        print(s)
