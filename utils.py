import torch
from config import OptConfig

GB = 1 << 30

class ValueHolder:
    def __init__(self):
        self.val = None

    def store(self, val):
        assert self.val is None
        self.val = val

    def pop(self):
        ret = self.val
        self.val = None
        return ret

    def clear(self):
        self.val = None


def array_1d(a, cls):
    return [cls() for _ in range(a)]


def array_2d(a, b, cls):
    return [[cls() for _ in range(b)] for _ in range(a)]


def array_3d(a, b, c, cls):
    return [[[cls() for _ in range(c)] for _ in range(b)] for _ in range(a)]


def array_4d(a, b, c, d, cls):
    return [[[[cls() for _ in range(d)] for _ in range(c)] for _ in range(b)] for _ in range(a)]

    
def prettyTime(seconds):
    # Calculate hours, minutes, and seconds
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    # Create a human-friendly string
    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours}h")
    if minutes > 0:
        time_parts.append(f"{minutes}m")
    if seconds > 0 or (hours == 0 and minutes == 0):
        time_parts.append(f"{seconds}s")
    
    return ":".join(time_parts)

# Example usage
# time_in_seconds = 3665  # 1 hour, 1 minute, and 5 seconds
# print(human_friendly_time(time_in_seconds))

def integerType(t):
    return t in {
        torch.int8, 
        torch.int16,
        torch.int32,
        torch.int64
    }

def elementSize(t):
    if t == torch.float16:
        return 2
    elif t == torch.float32:
        return 4
    elif t == torch.float64:
        return 8
    else:
        raise NotImplementedError(f"unsupported type: {t}")   

def peekTensor(t, prompt):
    print(f"{prompt} {t}")

def model_components_bytes(config: OptConfig):
    h = config.hiddenSize
    attentionElements = config.numHiddenLayer * h * (3 * h + 1) + h * (h + 1) 
    feedforwardElements = config.numHiddenLayer * h * (4 * h + 1) + h * 4 * (h + 1)
    othersElements = config.numHiddenLayer * h * 4 + config.vocabSize * (h + 1)

    return attentionElements * elementSize(config.dtype), \
           feedforwardElements * elementSize(config.dtype), \
           othersElements * elementSize(config.dtype)

    
def model_bytes(config: OptConfig):
    h = config.inputDim
    nelement = (config.numHiddenLayer * (
    # attention
    h * (3 * h + 1) + h * (h + 1) +
    # mlp
    h * (4 * h + 1) + h * 4 * (h + 1) +
    # layer norm
    h * 4) +
    # embedding
    config.vocabSize * (h + 1))

    return nelement * elementSize(config.dtype)

def cache_bytes(config:OptConfig, batchSize, seqLen):
    # assuming float16
    nelement = config.numHiddenLayer * seqLen * batchSize * config.inputDim * 2
    return nelement * elementSize(config.dtype)

def hidden_bytes(config:OptConfig, batchSize, seqLen):
    # assuming float16
    return batchSize * seqLen * config.inputDim * elementSize(config.dtype)

def report(banner=None, 
           model=None, 
           prefetch=None, 
           offload=None, 
           batchSize=None, 
           compress=None, 
           modelSize=None, 
           cacheSize=None, 
           hiddenSize=None, 
           loadTime=None, 
           inferenceTime=None, 
           perTokenTime=None,
           prefillTime=None,
           decodeTime=None,
           throughput=None):
    r = f"\n {'>>>'*6} {banner} {'<<<' * 6}\n" \
        if banner is not None \
        else ""

    for tag, p in zip(
        ["model", "prefetch", "offload", "batchSize", "compress", "loadTime", "inferenceTime" ],
        [model,    prefetch ,  offload,   batchSize,   compress,   loadTime,   inferenceTime ]):
        if p is not None:
            r += f" >>> {tag}: {p}\n"

    for tag, s in zip(
        ["modelSize", "cacheSize", "hiddenSize"],
        [modelSize, cacheSize, hiddenSize ]
    ):
        if s is not None:
            r += f" >>> {tag}: {s / GB:.3f}GB\n"
    
    for tag, t in zip(
        ["prefillTime", "per-token", "decodeTime"],
        [prefillTime, perTokenTime, decodeTime]
    ):
        if t is not None:
            r += f" >>> {tag}: {prettyTime(t)}\n"

    if throughput is not None:
        r += f" >>> throughput : {throughput:.4f} token/s\n"

    return r