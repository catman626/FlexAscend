from transformers import AutoTokenizer

from torchBackend import dtype, linear, layernorm, arange,  ReLU, Add, cumsum, BatchMatMul, Argmax, sqrt, makeMask, argmax

# from mindspore import nn, ops, dtype, Parameter, common, Tensor
# from mindspore.common.initializer import initializer, Zero
# from mindspore import load_checkpoint

from torchBackend import FlexTensor

import numpy as np
import os


from timer import timers
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import threading
from utils import ValueHolder, prettyTime

from optConfig import OptConfig, getOptConfig

class Linear:
    def __init__(self, name, inputChannel:int, outputChannel:int):
        self.name = name
        self.inchannel = inputChannel
        self.outchannel = outputChannel
        self.weight : FlexTensor = FlexTensor(name+".weight", (outputChannel, inputChannel))
        self.bias   : FlexTensor = FlexTensor(name+".bias", (outputChannel))

    def __call__(self, x:FlexTensor):
        return linear(x, self.weight, self.bias)

    def getParameters(self):
        return { self.weight, self.bias }

    def loadWeight(self):
        self.weight.load()
        self.bias.load()

class Layernorm:
    def __init__(self, name, normDim:int):
        self.name = name
        self.weight = FlexTensor(name+".weight", (normDim, ))
        self.bias   = FlexTensor(name+".bias", (normDim, ))
        self.normDim = normDim

    def __call__(self, x:FlexTensor):
        l = layernorm(normalized_shape=(self.normDim, ), 
                         weight=self.weight, 
                         bias=self.bias)
        return l(x)
    
    def getParameters(self):
        return { self.weight, self.bias }

    def loadWeight(self):
        self.weight.load()
        self.bias.load()

class Attention:
    #attn
    def __init__(self, name, config: OptConfig):
        super().__init__()
        self.config = config
        self.name = name
        self.headDim = config.hiddenSize // config.numHead
        self.seqLength = config.maxSeqLen
        hiddenSize = config.hiddenSize
        self.normFactor = sqrt(self.headDim)
        self.numHead = config.numHead
        
        self.qProj      = Linear(self.name+".qProj", hiddenSize, hiddenSize)
        self.kProj      = Linear(self.name+".kProj", hiddenSize, hiddenSize)
        self.vProj      = Linear(self.name+".vProj", hiddenSize, hiddenSize)
        self.outProj    = Linear(self.name+".outProj", hiddenSize, hiddenSize)
        self.layernorm  = Layernorm(name+".layernorm", normDim=hiddenSize)

    def getParameters(self):
        return {
            *self.kProj.getParameters(),
            *self.vProj.getParameters(),
            *self.qProj.getParameters(),
            *self.outProj.getParameters(),
            *self.layernorm.getParameters()
        }

    def loadWeight(self):
        for l in [ self.qProj, self.kProj, self.vProj, self.outProj, self.layernorm ]:
            l.loadWeight()

    def prefill(self, x, attentionMask):

        """ all in form (b, s, h)  """
        self.kCache = FlexTensor(self.name+".kcache", (b,self.config.maxSeqLen, h))
        self.vCache = FlexTensor(self.name+".vcache", (b,self.config.maxSeqLen, h))

        x = self.layernorm(x)

        return prefill(x, attentionMask, self.kCache, self.vCache)
    
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
        kcache = self.kCache.data()
        vcache = self.vCache.data()
        k = concat((kcache, k), axis=1)
        v = concat((vcache, v), axis=1)
        self.kCache.store(k)
        self.vCache.store(v)

        mhaOut = mha_decode(q, k, v, attentionMask, self.numHead)

        attnOut = self.outProj(mhaOut)

        attnOut = self.residual(attnOut, x)

        return attnOut

    def construct(self, x, iterno, attnMask):
        assert attnMask is None or len( attnMask.shape ) == 2

        if iterno == 0:
            return self.prefill(x, attnMask)
        else :
            return self.decode(x, attnMask)
        

class FeedForward:
    #ffn
    def __init__(self, name:str, config:OptConfig):
        super().__init__()
        self.name = name
        hiddenSize = config.hiddenSize
        ffnHiddenSize = config.ffnHiddenSize
        
        self.layernorm = Layernorm(name+".layernorm", normDim=hiddenSize)
        self.linear1 = Linear(name+".linear1", hiddenSize, ffnHiddenSize)
        self.relu = ReLU()
        self.linear2 = Linear(name+".linear2", ffnHiddenSize, hiddenSize)
        self.residual = Add()

    def getParameters(self):
        return {
            *self.layernorm.getParameters(),
            *self.linear1.getParameters(),
            *self.linear2.getParameters(),
        }
            
    def loadWeight(self):
        for l in [ self.layernorm, self.linear1, self.linear2 ]:
            l.loadWeight()

    def construct(self, x, iterno, attnMask):
        # assert x.dtype == dtype.float32

        o = self.layernorm(x)
        o = self.linear1(o)
        o = self.relu(o)
        o = self.linear2(o)
        
        ffnOut = self.residual(o, x)
        return ffnOut
        

class TransformerLayer:
    #transformerlayer
    
    def __init__(self, name, config:OptConfig):
        super().__init__()
        self.name = name
        self.attn = Attention(name=name+".attn", config=config)
        self.ffn = FeedForward(name=name+".ffn", config=config)

    def getParameters(self):
        return self.attn.getParameters().union(self.ffn.getParameters())

    def loadWeight(self):
        self.attn.loadWeight()
        self.ffn.loadWeight()

    def construct(self, x:FlexTensor, iterno, attnMask):
        assert x.dtype == dtype.float32  
        attnOut = self.attn(x, iterno, attnMask)
        
        ffnOut = self.ffn(attnOut, iterno, attnMask) 

        assert ffnOut.dtype==dtype.float32
        return ffnOut

def lazyParameter(shape, name):
    return Parameter(
            initializer(init="normal", shape = shape),
            name = name,
        )

class InputEmbed(nn.Cell):
    #inputembed
    def __init__(self, config:OptConfig):
        super().__init__()
        self.tokenEmbedWeight = FlexTensor(shape=(config.vocabSize, config.hiddenSize), name="inputEmbed.tokenWeight")
        self.posEmbedWeight = FlexTensor(shape=(config.maxSeqLen + 2, config.hiddenSize), name="inputEmbed.posWeight")
        self.gather = Gather()
        self.add = Add()

    def getParameters(self):
        return { self.tokenEmbedWeight, self.posEmbedWeight }


    def loadWeight(self):
        for l in [ self.tokenEmbedWeight, self.posEmbedWeight ]:
            l.load()

    def construct(self, inputIDs:FlexTensor, iterno, attentionMask:FlexTensor):
        """
        inputIDs : (B, S) / (B, 1)
        each element is an idx
        attentionMask: (B, S)
        """
        assert len(inputIDs.shape) == 2
        assert isinstance(inputIDs, FlexTensor)
        assert isinstance(attentionMask, FlexTensor)
        # assert self.tokenEmbedWeight.dtype == dtype.float32
        # assert self.posEmbedWeight.dtype == dtype.float32

        tokenEmbed = self.gather(self.tokenEmbedWeight, inputIDs, 0)

        posIds = cumsum(attentionMask.to(dtype=dtype.int32), axis=1)
        
        posEmbed = self.gather(self.posEmbedWeight, posIds, 0)

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

class OutputEmbed:
    #outputembed
    def __init__(self, config:OptConfig):
        super().__init__()
        self.tokenWeight = FlexTensor(shape=(config.vocabSize, config.hiddenSize), name="outputEmbed.tokenWeight")
        
        self.layernorm = Layernorm("outputEmbed.layernorm", config.hiddenSize)    
        self.matmul = BatchMatMul(transpose_b=True)
        self.argmax = Argmax()

    def getParameters(self):
        return self.layernorm.getParameters().union({self.tokenWeight})

    def loadWeight(self):
        self.tokenWeight.load()
        self.layernorm.loadWeight()
    
    def construct(self, x, iterno, attnMask):
        """
        (B, S, H) -> (B, )
        will take the last token to expect 
        """
        assert self.tokenWeight.data().dtype == dtype.float32, f"invalid dtype: {self.tokenWeight.dtype}"
        assert len(x.shape) == 3

        lastToken = x[:, -1, :]

        normalized = self.layernorm(lastToken)
        
        output = self.matmul(normalized, self.tokenWeight.data())
        # print(f">>> before argmax, output[0] is {output[0]}, shape: {output[0].shape}")  # (vocab)
        
        outputIDs = argmax(output)
        # print(f">>> after argmax, output[0] is {outputIDs[0]}, shape: {outputIDs[0].shape}")
        
        assert len(outputIDs.shape) == 1   # output shape: (B, S), element is id
        return outputIDs
    

    
class OPT:
    prefetch = False
    def __init__(self, config:OptConfig):
        super().__init__()
        self.config = config
        self.numLayers = config.numHiddenLayer  + 2
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, padding_side="left") 
        
        layers = []
        self.inputEmbed = InputEmbed(config)
        
        layers.append(self.inputEmbed)

        
        for i in range(config.numHiddenLayer):
            layers.append(
                TransformerLayer(f"layers.{i}", config = config)
            )
        self.outputEmbed = OutputEmbed(config)  
        layers.append(self.outputEmbed) 

        self.layers = layers
        
        self.loadModel(config.weightFname)

        self.tokensBuffer: FlexTensor = None
        self.maxSeqLen = config.maxSeqLen
        self.attentionMask :FlexTensor = None   # true : valid, false : neglect

    def getParameters(self):
        ret = set()
        for l in self.layers:
            ret = ret.union(l.getParameters())
        return ret
        
    def initZeros(self):
        for p in tqdm(self.getParameters()):
            p.initZeros()

    def loadModelFromFile(self, weightFname:str):
        assert isinstance(weightFname, str), f"invalid type: {type(weightFname)}"
        
        weights = load_checkpoint(weightFname)
        unusedWeight = set(weights.keys())
        loaded = []

        for p in self.getParameters():
            if p.name in weights.keys():
                p.store(weights[p.name].to(dtype.float32))
                unusedWeight.remove(p.name)
                loaded.append(p.name)

        if unusedWeight:
            print(f" !!! unused weight")
            for uw in unusedWeight:
                print(uw)

        return loaded


    def loadModel(self, weightFname):
        if not weightFname:
            print(f" >>> ckpt not provided")
            print(f" >>> use dummy weight")
            self.initZeros()
            return

        print(">>> load weight begin")
        loaded = []
        if isinstance(weightFname, str):
            ld = self.loadModelFromFile(weightFname)  
            loaded.append(ld)
        else:
            for w in weightFname:
                ld = self.loadModelFromFile(w)
                loaded.extend(ld)
                
        print("<<< load weight finish")
        
        fail = False
        for p in self.getParameters():
            if p.name not in loaded:
                print(f" !!! weight not loaded: {p.name}")
                fail = True
            
        if fail:
            print(" >>> successfully loaded: ")
            for l in loaded:
                print(l)
            exit(1)


    def compute(self, s, l):
        h = self.layers[l](self.hidden[l].val, s, self.attentionMask) 
        self.hidden[l+1].store(h) 

    def loadLayer(self, l):
        if l < 0 or l >= self.numLayers:
            return 
        
        self.layers[l].loadWeight()
        
    def coreLoop(self,  iterNo):
        if OPT.prefetch: 
            # raise NotImplementedError("!!! not implemented") 
            self.loadLayer(0)
            for l in range(self.numLayers):
                t1 = threading.Thread(target=self.loadLayer, args=[l+1])
                t2 = threading.Thread(target=self.compute, args=[iterNo, l])

                t2.start()
                t1.start()
                
                t1.join()
                t2.join()
            return 

        for l in range(self.numLayers):
            # print(f" \t\t>>>layer {l}")
            self.loadLayer(l)
            self.compute(iterNo, l)

    def singleToken(self, i, currLen):
        B = self.tokensBuffer.shape[0]
        self.hidden = [ ValueHolder() for _ in range(self.numLayers+1)]
        
        inputIDs = self.tokensBuffer[:, :currLen] if i == 0 \
            else self.tokensBuffer[:, -1:]
        
        self.hidden[0].store(inputIDs)
        
        self.coreLoop(i)
        
        self.tokensBuffer = concat([self.tokensBuffer, self.hidden[-1].val.unsqueeze(1)], axis=1)
        self.attentionMask = concat(
            (self.attentionMask, FlexTensor(np.ones(shape=(B,  1), dtype=np.bool_)) ), 
            axis=1)
            
    def run(self, inputSentences: list[str]):
        promptLen = max([len(l) for l in inputSentences])
        inputTokens = self.tokenizer(inputSentences, padding="max_length",  max_length=promptLen).input_ids
        self.tokensBuffer = FlexTensor(inputTokens, dtype=dtype.int32) 

        maxIter = 20 
        
        # init attention mask
        self.attentionMask = (self.tokensBuffer != self.config.padTokenID)
        
        print(">>> inference begin")
        for i in range(maxIter):
            print(f"    >>> loop {i} begin")
            self.singleToken(i, promptLen+i)
            
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
    
    parser.add_argument("--ckpt", nargs="*", help="list all ckpt files")
    parser.add_argument("--tokenizer", type=str, default="/home/ma-user/work/FlexAscend/model/opt-1.3b")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--interact", action="store_true")

    args = parser.parse_args()

    config = getOptConfig(args.model)
    config.weightFname = args.ckpt
    config.tokenizer = args.tokenizer

    if args.offload:
        FlexTensor.offload = True
    if args.prefetch:
        OPT.prefetch = True
    if args.compress:
        FlexTensor.setCompess(True)

    print("\n " + ">>>"*6 + " inference begin " + "<<<"*6)
    print(f" >>> settings: ")
    print(f" >>> model: {args.model}")
    print(f" >>> prefetch: {OPT.prefetch}")
    print(f" >>> offload: {args.offload}")
    timers("load").start()
    model = OPT(config)
    timers("load").stop()

    testBatchSize = 64
    inputs = [
        "Beijing is the capital city of",
    ] * testBatchSize
    
    print(f" >>> batch size: {testBatchSize}")
    timers("model").start()
    
    outputs = model.run(inputs)
    for s in outputs:
        print(s)

    timers("model").stop()

    inferenceTime = timers("model").elapsed()
    loadTime = timers("load").elapsed()

    if args.interact:
        while True:
            sentence = input(" >>> plase input the question\n")
            if sentence == "xxx":
                break
            outputs = model.run([sentence])
            for s in outputs:
                print(s)

    print(f" >>> load model take time: {prettyTime(loadTime)}")
    print(f" >>> inference take time: {prettyTime(inferenceTime)}")

    with open("default_log", "a+") as f:
        f.write(f"\n {'>>>'*6} model run {'<<<' * 6}")
        f.write(f" >>> model: {args.model}\n")
        f.write(f" >>> prefetch: {OPT.prefetch}\n")
        f.write(f" >>> offload: {args.offload}\n")
        f.write(f" >>> batch size: {testBatchSize}")
        f.write(f" >>> compress: {args.compress}")
        f.write(f" >>> load model take time: {prettyTime(loadTime)}\n")
        f.write(f" >>> inference take time: {prettyTime(inferenceTime)}\n")
