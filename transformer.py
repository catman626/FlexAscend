import abc
import numpy as np
import mindspore as ms
from mindspore import nn
import math
from mindformers import LlamaTokenizer


class Config:
    def __init__(self):
        self.dtype = ms.dtype.float16
        self.hiddenSize = 512
        self.ffnHiddenSize = 4 * self.hiddenSize
        self.numHeads = 8
        self.hasBias = True
        self.seqLength = 256
        self.batchSize = 64

        self.numLayers = 32

config = Config()


class Attention(nn.Cell):
    def __init__(self, config: Config):
        super().__init__()

        self.headDim = config.hiddenSize // config.numHeads

        self.normFactor = math.sqrt(self.headDim)
        self.q = nn.Linear(in_features=config.hiddenSize, 
                           out_features=config.hiddenSize,
                           weight_init="normal",
                           bias=config.hasBias)
        
        self.k = nn.Linear(in_features=config.hiddenSize, 
                           out_features=config.hiddenSize,
                           weight_init="normal",
                           bias=config.hasBias)

        self.v = nn.Linear(in_features=config.hiddenSize, 
                           out_features=config.hiddenSize,
                           weight_init="normal",
                           bias=config.hasBias)

        self.out = nn.Linear(in_features=config.hiddenSize,
                             out_features=config.hiddenSize,
                             weight_init="normal",
                             dtype=config.dtype)

        self.batchMatMul = ms.ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax()
        self.batchMatMulSV = ms.ops.BatchMatMul()
        
        self.rawAttention = nn.MultiheadAttention(embed_dim=config.hiddenSize,
                                                  num_heads=config.numHeads)
        
    def back_construct(self, x, mask):
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        
        score = self.batchMatMul(query, key)

        normalized = score / self.normFactor
        
        softmax = self.softmax(normalized)

        out = self.batchMatMulSV(softmax, value)
        
        return softmax

    def construct(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)

        attnOutput, attnOutputWeight = self.rawAttention(q, k, v)

        

        return attnOutput
    

        

model = Attention(config=config)
inputShape = (config.batchSize, config.seqLength, config.hiddenSize)
inputSeq = ms.Tensor(np.random.normal(loc=0, scale=0.01, size=inputShape), dtype=ms.float32)
mask = ms.numpy.ones((config.batchSize, config.seqLength, config.seqLength))
output = model(inputSeq)

print(output.shape)

class FeedForward(nn.Cell):
    def __init__(self, config:Config):
        super().__init__()

        self.layers = nn.SequentialCell(
            nn.Linear(in_features=config.hiddenSize,
                    out_features=config.ffnHiddenSize,
                    bias=True),
            nn.GELU(),
            nn.Linear(in_features=config.ffnHiddenSize,
                      out_features=config.hiddenSize,
                      bias=True),
            nn.GELU()
        )

    def construct(self, x):
        return self.layers(x)

class TransformerLayer(nn.Cell):
    
    def __init__(self, config:Config):
        super().__init__()
        self.attn = Attention(config)
        self.ffn = FeedForward(config)

        self.add1 = ms.ops.Add()
        self.norm1 = nn.LayerNorm(normalized_shape=(config.hiddenSize, ))
        
        self.add2 = ms.ops.Add()
        self.norm2 = nn.LayerNorm(normalized_shape=(config.hiddenSize, ))


    def construct(self, x):
        attnOut = self.attn(x)
        
        norm1 = self.norm1(self.add1(attnOut, x))
        
        ffnOut = self.ffn(norm1) 

        norm2 = self.norm2(self.add2(ffnOut, norm1)) 

        return norm2
        
   
class Layer(nn.Cell):
    @abc.abstractmethod
    def initialized(self):
        pass


class Llama(nn.Cell):
    pass    

print(output.shape)
model = TransformerLayer(config=config)
inputShape = (config.batchSize, config.seqLength, config.hiddenSize)
inputSeq = ms.Tensor(np.random.normal(loc=0, scale=0.01, size=inputShape), dtype=ms.float32)
mask = ms.numpy.ones((config.batchSize, config.seqLength, config.seqLength))
output = model(inputSeq)

class LlamaInputEmbed(Layer):
    def __init__(self, name, config):
        self.embedding = self.initEmbedWeight(config.embeddingFileName)
        self.name = name
        self.gather = ms.ops.operations.Gather()

    def construct(self, inputIDs):
        if self.embedding is None:
            raise ValueError(f"embed {self.name} is not initialized!")
        return self.gather(input_params=self.embedding, 
                    input_indices=inputIDs,
                    axis=0)

       
class LlamaOutputEmbed(Layer):
    
    
    
class Llama(nn.Cell):
    def __init__(self, config:Config):
        self.numLayers = config.numLayers
        
        layers = []
        self.tokenizer = LlamaTokenizer.from_pretrained("llama_7b")
        
        layers.append(LlamaInputEmbed(config))
        for i in config.numLayers:
            layers.append(
                TransformerLayer(name = f"transformerlayer-{i}", config = config)
            )
        layers.append(LlamaOutputEmbed(config))  
        
        self.layers = layers
        
        self.loadWeight(config.weightFnames)
        

    def loadWeight(self, weightFnames):
        if isinstance(weightFnames, list):
            for n in weightFnames:
                self.loadWeight(n)
        
        assert isinstance(weightFnames, str)
        
        weights = ms.load_checkpoint(weightFnames)
        
        for l in self.layers:
            if not l.initialized():
                if l.name in weights:
                    l.weight = weights[l.name]


        