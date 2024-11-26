from .transformer import config, Attention
import mindspore as ms
import numpy as np


model = Attention(config=config)
inputShape = (config.batchSize, config.seqLength, config.hiddenSize)
inputSeq = ms.Tensor(np.random.normal(loc=0, scale=0.01, size=inputShape), dtype=ms.float32)
mask = ms.numpy.ones((config.batchSize, config.seqLength, config.seqLength))
output = model(inputSeq)

print(output.shape)