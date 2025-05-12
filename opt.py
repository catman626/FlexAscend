from transformers import AutoTokenizer

import torch
import torch.nn.functional as F
from torch import Tensor

from backend import FlexTensor, mha_prefill, mha_decode

import os
from timer import timers
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import threading
from utils import ValueHolder, prettyTime, integerType, peekTensor, GB
import utils 

from config import OptConfig, getOptConfig

cnt = 0

class Linear:
    def __init__(self, name, inputChannel:int, outputChannel:int, weightHome:str):
        self.name = name
        self.inchannel = inputChannel
        self.outchannel = outputChannel
        self.weightHome = weightHome
        self.weight : FlexTensor = FlexTensor(name+".weight", (outputChannel, inputChannel), weightHome)
        self.bias   : FlexTensor = FlexTensor(name+".bias", (outputChannel, ), weightHome)

    def __call__(self, x:FlexTensor):
        w = self.weight.data().to("cuda")
        b = self.bias.data().to("cuda") 
        linearOut = F.linear(x, w, b)
        del w, b
    
        return linearOut

    def getParameters(self):
        return { self.weight, self.bias }

    def loadWeight(self):
        self.weight.load()
        self.bias.load()

class Layernorm:
    def __init__(self, name, normDim:int, computeDev, weightHome):
        self.name = name
        self.weight = FlexTensor(name+".weight", (normDim, ), weightHome)
        self.bias   = FlexTensor(name+".bias", (normDim, ), weightHome)
        self.normDim = normDim
        self.computeDev = computeDev
        assert computeDev in ["CPU", "GPU"], f"invalid computeDev: {computeDev}"

    def __call__(self, x:Tensor):
        w = self.weight.data()
        b = self.bias.data()
        if self.computeDev == "GPU":
            x = x.to("cuda")
            w = w.to("cuda")
            b = b.to("cuda")
        
        lnOut = F.layer_norm(input=x, 
                         normalized_shape=(self.normDim, ), 
                         weight=w, 
                         bias=b)

        del w, b
        return lnOut
    
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
        self.numHead = config.numHead
        self.weightHome = "CPU" \
            if "w" in config.offload \
            else "GPU"
        self.cacheHome = "CPU" \
            if "c" in config.offload \
            else "GPU"
        
        self.qProj      = Linear(self.name+".qProj", 
                                 hiddenSize, hiddenSize, 
                                 self.weightHome)
        self.kProj      = Linear(self.name+".kProj", 
                                 hiddenSize, hiddenSize, 
                                 self.weightHome)
        self.vProj      = Linear(self.name+".vProj", 
                                 hiddenSize, hiddenSize, 
                                 self.weightHome)
        self.outProj    = Linear(self.name+".outProj", 
                                 hiddenSize, hiddenSize, 
                                 self.weightHome)
        self.layernorm  = Layernorm(name+".layernorm", 
                                    normDim=hiddenSize,
                                    computeDev="GPU",
                                    weightHome=self.weightHome)

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

    def prefill(self, x:Tensor, attentionMask:Tensor):
        """ 
        x: (b, s, h)  
        attentionMask in shape: (B, s)
        """
        global cnt
        
        attentionMask = attentionMask.to("cuda")

        b, s, h = x.shape
        self.kCache = FlexTensor(self.name+".kcache", 
                                 (b,self.config.maxSeqLen, h),
                                 self.cacheHome)
        self.vCache = FlexTensor(self.name+".vcache", 
                                 (b,self.config.maxSeqLen, h),
                                 self.cacheHome)

        normalX = self.layernorm(x)

        q, k, v = self.qProj(normalX), self.kProj(normalX), self.vProj(normalX)
        self.kCache.store(k)
        self.vCache.store(v)

        # make a casual mask and combine it with attention mask
        mhaOut = mha_prefill(q, k, v, attentionMask, self.numHead) 

        attnOut = self.outProj(mhaOut)

        attnOut = attnOut + x
        
        cnt = cnt+1

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
        kcache = self.kCache.data().to("cuda")
        vcache = self.vCache.data().to("cuda")
        k = torch.concat([kcache, k], axis=1)
        v = torch.concat([vcache, v], axis=1)
        self.kCache.store(k)
        self.vCache.store(v)

        attentionMask = attentionMask.to("cuda")

        mhaOut = mha_decode(q, k, v, attentionMask, self.numHead)

        attnOut = self.outProj(mhaOut)

        attnOut = attnOut + x

        return attnOut
        

    def __call__(self, x, iterno, attnMask):
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
        weightHome = "CPU" \
            if "w" in config.offload \
            else "GPU"
        
        self.layernorm = Layernorm(name+".layernorm", 
                                   normDim=hiddenSize, 
                                   computeDev="GPU", 
                                   weightHome=weightHome)

        self.linear1 = Linear(name+".linear1", hiddenSize, ffnHiddenSize, weightHome)
        self.relu = torch.nn.ReLU()
        self.linear2 = Linear(name+".linear2", ffnHiddenSize, hiddenSize, weightHome)

    def getParameters(self):
        return {
            *self.layernorm.getParameters(),
            *self.linear1.getParameters(),
            *self.linear2.getParameters(),
        }
            
    def loadWeight(self):
        for l in [ self.layernorm, self.linear1, self.linear2 ]:
            l.loadWeight()

    def __call__(self, x, iterno, attnMask):
        o = self.layernorm(x)
        o = self.linear1(o)
        o = self.relu(o)
        o = self.linear2(o)
        ffnOut = o + x
        return ffnOut
        

class TransformerLayer:
    #transformerlayer
    def __init__(self, name, config:OptConfig):
        super().__init__()
        self.name = name
        self.attn = Attention(name=name+".attn", config=config)
        self.ffn = FeedForward(name=name+".ffn", config=config)

    def getParameters(self):
        return {
            *self.attn.getParameters(),
            *self.ffn.getParameters()
        }

    def loadWeight(self):
        self.attn.loadWeight()
        self.ffn.loadWeight()

    def __call__(self, x:FlexTensor, iterno, attnMask):
        attnOut = self.attn(x, iterno, attnMask)
        
        ffnOut = self.ffn(attnOut, iterno, attnMask) 

        return ffnOut


class InputEmbed:
    #inputembed
    def __init__(self, config:OptConfig):
        super().__init__()
        self.tensorHome = "CPU" \
            if "w" in config.offload \
            else "GPU"
            
        self.tokenEmbedWeight = FlexTensor("inputEmbed.tokenWeight", 
                                           (config.vocabSize, config.hiddenSize), "CPU")
        self.posEmbedWeight = FlexTensor("inputEmbed.posWeight",
                                         (config.maxSeqLen + 2, config.hiddenSize),
                                         "CPU")

    def getParameters(self):
        return { self.tokenEmbedWeight, self.posEmbedWeight }

    def loadWeight(self):
        for l in self.getParameters():
            l.load()

    def __call__(self, inputIDs:Tensor, iterno, attentionMask:Tensor):
        """
        inputIDs : (B, S) / (B, 1)
        each element is an idx
        attentionMask: (B, S)
        """
        assert len(inputIDs.shape) == 2
        assert isinstance(inputIDs, Tensor)
        assert integerType(inputIDs.dtype), f"embedding input not integer: {inputIDs.dtype}"
        assert inputIDs.device.type == "cpu"
        assert self.posEmbedWeight.data().device.type == "cpu"
        assert self.tokenEmbedWeight.data().device.type == "cpu"
        assert inputIDs.device.type == "cpu"

        tokenEmbed = F.embedding( inputIDs, self.tokenEmbedWeight.data() ,0)

        posIds = torch.cumsum(attentionMask, dim=1) * attentionMask + 1
        
        posEmbed = F.embedding(posIds, self.posEmbedWeight.data())
        
        currLength = attentionMask.shape[1]
        inputIDLength = inputIDs.shape[1]
        previousIDsLength = currLength - inputIDLength
        posEmbed = posEmbed[:, previousIDsLength:]

        assert tokenEmbed.shape == posEmbed.shape
        embed = tokenEmbed + posEmbed

        embed = embed.to("cuda")
        
        
        assert len(embed.shape) == 3
        return embed

class OutputEmbed:
    #outputembed
    def __init__(self, config:OptConfig):
        super().__init__()
        # self.tensorHome = "CPU" \
        #     if "w" in config.offload \
        #     else "GPU"
            
        self.tokenWeight = FlexTensor(shape=(config.vocabSize, config.hiddenSize), 
                                    name="outputEmbed.tokenWeight", 
                                    home="CPU")
        
        self.layernorm = Layernorm("outputEmbed.layernorm", 
                                   config.hiddenSize, 
                                   computeDev="CPU",
                                   weightHome="CPU")    

    def getParameters(self):
        return {
            *self.layernorm.getParameters(),
            self.tokenWeight
        }

    def loadWeight(self):
        self.tokenWeight.load()
        self.layernorm.loadWeight()
    
    def __call__(self, x, iterno, attnMask):
        """
        (B, S, H) -> (B, )
        will take the last token to expect 
        """
        assert len(x.shape) == 3
        
        
        if x.device.type != "cpu":
            x = x.cpu() 
        
        lastToken = x[:, -1, :]

        normalized = self.layernorm(lastToken)
        
        output = F.linear(normalized, self.tokenWeight.data())

        outputIDs = torch.argmax(output, dim=-1)
        
        assert len(outputIDs.shape) == 1   # output shape: (B, S), element is id
        assert integerType(outputIDs.dtype)
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
        
        weights = torch.load(weightFname)
        unusedWeight = set(weights.keys())
        loaded = []

        for p in self.getParameters():
            if p.name in weights.keys():
                p.store(weights[p.name].to(torch.float32))
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
        
    def coreLoops(self,  iterNo):
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
            self.loadLayer(l)
            self.compute(iterNo, l)

        # if iterNo == 0:
        #     for l in range(self.numLayers + 1):
        #         torch.save(self.hidden[l].val, f"comp/my/{l}")

    def singleToken(self, i, currLen):
        B = self.tokensBuffer.shape[0]
        self.hidden = [ ValueHolder() for _ in range(self.numLayers+1)]
        
        inputIDs = self.tokensBuffer[:, :currLen] if i == 0 \
            else self.tokensBuffer[:, -1:]
        
        self.hidden[0].store(inputIDs)
        
        if i == 0:
            timers("prefill").start()   
            self.coreLoops(i)
            timers("prefill").stop()
        else:
            self.coreLoops(i)
        
        self.tokensBuffer = torch.concat([self.tokensBuffer, self.hidden[-1].val.unsqueeze(1)], axis=1)
        self.attentionMask = torch.concat(
            [self.attentionMask, torch.ones((B,  1), dtype=torch.bool) ], dim=1)
            
    def run(self, inputSentences: list[str], numIter):
        # promptLen = max([len(l) for l in inputSentences])
        inputTokens = self.tokenizer(inputSentences, padding="max_length", max_length=32).input_ids
        print(f"size of inputTokens: { [ len(l) for l in inputTokens ] }")
        self.tokensBuffer = Tensor(inputTokens).to(torch.int32) 
        
        promptLen = self.tokensBuffer.shape[1]

        # init attention mask
        self.attentionMask = (self.tokensBuffer != self.config.padTokenID)
        
        print(" >>> inference begin")
        for i in range(numIter):
            print(f" \t>>> inference iter {i+1}/{numIter}")
            self.singleToken(i, promptLen+i)
            
        print(" <<< inference end")
        outputSentences = []
        
        for line in self.tokensBuffer.tolist():
            sentence = self.tokenizer.convert_ids_to_tokens(line)
            sentence = self.tokenizer.convert_tokens_to_string(sentence)

            outputSentences.append(sentence)
         
        return outputSentences

def getInputs(inputfile, batchSize):
    if inputfile is None:
        testPrompt = "Beijing is the capital city of"
        testPrompt = "in my palm is a clear stone , and inside it is a small ivory statuette . a guardian angel . `` figured if you 're going to be out at night getting hit by cars , you might as well have some backup . '' i look at him , feeling stunned . like this is some sort of sign . but as i stare at harlin , his mouth curved in a confident grin , i do n't care about"
        return [[ testPrompt] * batchSize ], len(testPrompt.split()) * 1.4

    with open(inputfile) as f:
        lines = f.readlines()
        inputs = [ lines[i:i+batchSize] for i in range(0, len(lines), batchSize)]

        estimatedPromptLen = max([ len(l) for l in lines ]) * 1.4
        
    return inputs , estimatedPromptLen

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ckpt", nargs="*", help="list all ckpt files")
    parser.add_argument("--tokenizer", type=str, default="/home/ma-user/work/FlexAscend/model/opt-1.3b")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--offload", default="", help="eg. "
                        "--offload w to offload weightm "
                        "--offload wc to offload weight and cache, ")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--interact", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--logfile", type=str, default="defaultLog")
    parser.add_argument("--gen-len", type=int, default=8)
    parser.add_argument("--response-file", type=str)
    parser.add_argument("--input-file", type=str)

    args = parser.parse_args()

    config = getOptConfig(args.model)
    config.weightFname = args.ckpt
    config.tokenizer = args.tokenizer
    config.offload = args.offload
    batchSize = args.batch_size

    logFile = args.logfile

    if args.prefetch:
        OPT.prefetch = True
    if args.compress:
        FlexTensor.setCompress(True)

    numIter = args.gen_len
    
    inputs, estimatedPromptLen = getInputs(args.input_file, batchSize)
    
    modelSize = utils.model_bytes(config)  
    cacheSize = utils.cache_bytes(config, batchSize, estimatedPromptLen + numIter)
    hiddenSize = utils.hidden_bytes(config, batchSize, estimatedPromptLen +numIter)

    print(utils.report(banner="inference begin",
                     model=config.modelName,
                     prefetch=args.prefetch,
                     offload=args.offload,
                     batchSize=batchSize,
                     compress=args.compress,
                     modelSize=modelSize,
                     cacheSize=cacheSize,
                     hiddenSize=hiddenSize
                     ))
    timers("load").start()
    model = OPT(config)
    timers("load").stop()
    
    timers("model").start()
    if not args.interact:
        for b in inputs:
            outputs = model.run(b, numIter)
            if args.ckpt is not None:
                if args.response_file is not None:
                    with open(args.response_file, "w+") as f:
                        respones = ("#" * 6 + "\n").join(outputs)
                        f.write(respones)
                else:
                    for s in outputs:
                        print(s)

    timers("model").stop()

    inferenceTime = timers("model").elapsed()
    loadTime = timers("load").elapsed()

    if args.interact:
        while True:
            sentence = input(" >>> plase input the question\n"
                             ">>> xxx to quit\n")
            if sentence == "xxx":
                break
            outputs = model.run([sentence], 32, 32)
            for s in outputs:
                print(s)

    prefillTime = timers("prefill").elapsed() / batchSize
    decodeTime = (timers("prefill").elapsed() - prefillTime) / batchSize
    throughtput =  batchSize * numIter / inferenceTime 
    print(utils.report(inferenceTime=inferenceTime,loadTime=loadTime, batchSize=args.batch_size, prefillTime=prefillTime, decodeTime=decodeTime, throughput=throughtput))

    with open(logFile, "a+") as f:
        r = utils.report(banner="OPT run",
                         model=config.modelName, 
                         prefetch=args.prefetch, 
                             offload=args.offload, 
                             batchSize=batchSize,
                             compress=args.compress,
                            modelSize=modelSize,
                            cacheSize=cacheSize,
                            hiddenSize=hiddenSize,
                            loadTime=loadTime,
                            inferenceTime=inferenceTime,
                            prefillTime=prefillTime,
                            throughput=throughtput
        )
        f.write(r)

    FlexTensor.clear()
