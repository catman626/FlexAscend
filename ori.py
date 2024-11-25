from mindspore import nn, ops
import mindspore as ms
import math
import numpy as np

class Config:
    
    def __init__(self):
        self.hidden_size = 512
        self.bias = True
        self.dtype = ms.float32
        self.num_heads = 4

        self.ffn_hidden_size = 4 * self.hidden_size
        self.seq_length = 1

config = Config()


class Attention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.num_heads_per_partition = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.norm_factor = math.sqrt(self.head_dim)
        self.q = nn.Linear(in_features=config.hidden_size,
                                      out_features=config.hidden_size,
                                      weight_init='normal',
                                      bias=config.bias)
                        
        self.k = nn.Linear(in_features=config.hidden_size,
                                      out_features=config.hidden_size,
                                      weight_init='normal',
                                      dtype=config.dtype,
                                      bias=config.bias)
        self.v = nn.Linear(in_features=config.hidden_size,
                                      out_features=config.hidden_size,
                                      weight_init='normal',
                                      dtype=config.dtype,
                                      bias=config.bias)
        self.flash_attention = ops.operations.nn_ops.FlashAttentionScore(head_num=self.num_heads_per_partition,
                                                                        scale_value=1.0/self.norm_factor,
                                                                        next_tokens=0)
        self.out = nn.Linear(in_features=config.hidden_size,
                                     out_features=config.hidden_size,
                                     weight_init='normal',
                                     dtype=config.dtype,
                                     bias=config.bias)

    def construct(self, x, mask):
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        _, _, _, context_layer = self.flash_attention(query, key, value, attn_mask=mask)
        output = self.out(context_layer)
        return output


class MLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(in_features=config.hidden_size,
                                       out_features=config.ffn_hidden_size,
                                       weight_init='normal',
                                       dtype=config.dtype,
                                       bias=config.bias)
        self.w2 = nn.Linear(in_features=config.ffn_hidden_size,
                                    out_features=config.hidden_size,
                                    weight_init='normal',
                                    dtype=config.dtype,
                                    bias=config.bias)
        self.act_func = nn.SiLU()
        self.mul = ops.Mul()

    def construct(self, x):
        x = self.w1(x)
        x = self.act_func(x)
        output = self.w2(x)
        return output

mask = np.ones(shape=(1, config.seq_length, config.seq_length), dtype=np.uint8)
mask = ms.Tensor(mask)
attention = Attention(config=config)
attention_output = attention(ms.Tensor(np.ones((1, 1, 512), dtype=np.float32)), mask)
print(attention_output.shape)

mlp = MLP(config=config)
mlp_output = mlp(attention_output)
print(mlp_output.shape)
