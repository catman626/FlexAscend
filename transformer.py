from mindspore import nn
import math



class Attention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.num_heads_per_partition = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.norm_factor = math.sqrt(self.head_dim)
        self.q = nn.Linear(in_channels=config.hidden_size,
                                      out_channels=config.hidden_size,
                                      weight_init='normal',
                                      has_bias=config.has_bias)
        self.k = nn.Linear(in_channels=config.hidden_size,
                                      out_channels=config.hidden_size,
                                      weight_init='normal',
                                      dtype=config.dtype,
                                      has_bias=config.has_bias)
        self.v = nn.Linear(in_channels=config.hidden_size,
                                      out_channels=config.hidden_size,
                                      weight_init='normal',
                                      dtype=config.dtype,
                                      has_bias=config.has_bias)
        self.flash_attention = ops.operations.nn_ops.FlashAttentionScore(head_num=self.num_heads_per_partition,
                                                                        scale_value=1.0/self.norm_factor,
                                                                        next_tokens=0)
        self.out = nn.Linear(in_channels=config.hidden_size,
                                     out_channels=config.hidden_size,
                                     weight_init='normal',
                                     dtype=config.dtype,
                                     has_bias=config.has_bias)

    def construct(self, x, mask):
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        _, _, _, context_layer = self.flash_attention(query, key, value, attn_mask=mask)
        output = self.out(context_layer)
        return output
