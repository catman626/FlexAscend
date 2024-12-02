from transformers import AutoTokenizer
from mindspore import nn, ops, dtype, Parameter, common, Tensor
from mindspore.common.initializer import initializer
from mindspore import load_checkpoint
import math
import abc
import numpy as np


import argparse
from tqdm import tqdm

class Config:
    def __init__(self):
        self.dtype = dtype.float16
        self.hiddenSize = 2048
        self.ffnHiddenSize = 4 * self.hiddenSize
        self.numHeads = 32
        self.hasBias = True
        self.seqLength = 2048
        self.batchSize = 64

        self.numHiddenLayer = 24
        self.vocabSize = 50272
        self.weightFname = None
        self.localTokenizer = "opt-1.3b"

config = Config()


class Attention(nn.Cell):
    def __init__(self, config: Config):
        super().__init__()

        self.headDim = headDim = config.hiddenSize // config.numHeads
        self.seqLength = config.seqLength
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
    
        
    def construct(self, x, iterNo, attentionMask=None):
        """ all in form (b, s, h)  """
        b, s, h = x.shape

        normalX = self.attnLayerNorm(x)
        # (b, s, h)
        q, k, v = self.qProj(normalX), self.kProj(normalX), self.vProj(normalX)

        if iterNo == 0:
            self.kCache = Tensor(np.zeros((b, self.seqLength, h)))
            self.vCache = Tensor(np.zeros((b, self.seqLength, h)))
            self.kCache[:, :s] = k
            self.vCache[:, :s] = v
            
        else :
            self.kCache[:, s] = k
            self.vCache[:, s] = v
            k = self.kCache[:, :s]
            v = self.vCache[:, :s]

        # (b, s, nh, h1)
        q = q.view(b, s, self.numHeads, self.headDim)
        k = k.view(b, s, self.numHeads, self.headDim)
        v = v.view(b, s, self.numHeads, self.headDim)

        q = ops.permute(q, (0, 2, 1, 3))
        k = ops.permute(k, (0, 2, 3, 1))
        v = ops.permute(v, (0, 2, 1, 3))
        # output shape (b, nh, s, s)
        score = self.batchMatMul(q, k)

        ids = ops.arange(end=s)
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
        attnOut = ops.permute(attnOut, (1, 2)).flatten(start_dim=2)
        
        attnOut = self.outProj(attnOut)

        attnOut = self.residual(attnOut, x)

        return attnOut
        

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

    def construct(self, x):
        attnOut = self.attn(x, iterNo=0)
        
        ffnOut = self.ffn(attnOut) 

        return ffnOut
        
   
class Layer(nn.Cell):
    @abc.abstractmethod
    def initialized(self):
        pass


def lazyParameter(shape, name):
    return Parameter(
            initializer(init="normal", shape = shape),
            name = name
        )

class OPTInputEmbed(Layer):
    def __init__(self, config:Config):
        super().__init__()
        self.tokenEmbedWeight = lazyParameter(shape=(config.vocabSize, config.hiddenSize), name="embed_tokens.weight")
        self.posEmbedWeight = lazyParameter(shape=(config.seqLength + 2, config.hiddenSize), name="embed_positions.weight")
        self.gather = ops.operations.Gather()
        self.add = ops.Add()

    def construct(self, inputIDs):
        tokenEmbed = self.gather(self.tokenEmbedWeight, inputIDs, 0)
        posEmbed = self.gather(self.posEmbedWeight, inputIDs, 0)
        embed = self.add(tokenEmbed, posEmbed)
        return embed
        


class OPTOutputEmbed(Layer):
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
        
        return self.argmax(output)
    
    
class OPT(nn.Cell):
    def __init__(self, config:Config):
        super().__init__()
        self.numLayers = config.numHiddenLayer
        
        self.tokenizer = AutoTokenizer.from_pretrained("/home/ma-user/work/FlexAscend/model/opt-1.3b") 
        
        layers = nn.SequentialCell()
        self.inputEmbed = OPTInputEmbed(config)
        for i in range(config.numHiddenLayer):
            layers.append(
                TransformerLayer(config = config)
            )
        self.outputEmbed = OPTOutputEmbed(config)  
        
        self.layers = layers
        
        self.loadWeight(config.weightFname)
        

    def loadWeight(self, weightFname):
        assert isinstance(weightFname, str)
        
        print("load weight begin")
        weights = load_checkpoint(weightFname)
        
        uninitializedInNet = []
        unusedWeight = set(weights.keys())
        for name, param in tqdm(self.parameters_and_names()):
            if name not in weights.keys():
                uninitializedInNet.append(name)
            else:
                param.set_data(weights[name])
                unusedWeight.remove(name)
                

        print("load weight finish")
        if uninitializedInNet:
            print(">>> uninitialized weight: ")
            for name in uninitializedInNet:
                print(name)

        if unusedWeight:
            print(">>> unused weight:") 
            for name in unusedWeight:
                print(name)

    def runIter(self, i):
        for l in self.layers:
            x = l(x) 

        return x
        
            
    def run(self, inputSentences: list[str]):
        inputTokens = self.tokenizer(inputSentences, padding="max_length",  max_length=config.seqLength).input_ids
        
        inputTokens = Tensor(inputTokens)
        h = self.inputEmbed(inputTokens)
        for l in self.layers:
            h = l(h) 
        outputIDs = self.outputEmbed(h)
        outputTokens = self.tokenizer.convert_ids_to_tokens(outputIDs)
        outputSentences = self.tokenizer.convert_tokens_to_string(outputTokens)

        return outputSentences
        
            
            
         

       
parser = argparse.ArgumentParser() 
parser.add_argument("--ckpt", type=str, default="model-weight/mindspore-weight.ckpt")
args = parser.parse_args()

config.weightFname = args.ckpt

model = OPT(config)


inputs = [
    "hello world!",
    "the largest cat in the world is "
]
model.run(inputs)
