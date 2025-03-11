from mindspore import Tensor, dtype, ops 

def mha_prefill(q:Tensor, k:Tensor, v:Tensor, mask:Tensor, numHead:int):
    assert q.dtype == dtype.float32
    assert k.dtype == dtype.float32
    assert v.dtype == dtype.float32

    b, s, h = q.shape
    
    assert h % numHead == 0
    headDim = h // numHead 

    scaling = headDim ** -0.5
    # (b, s, nh, h1)
    q = q.view(b, s, numHead, headDim) * scaling
    k = k.view(b, s, numHead, headDim)
    v = v.view(b, s, numHead, headDim)

    q = ops.permute(q, (0, 2, 1, 3))
    k = ops.permute(k, (0, 2, 3, 1))
    v = ops.permute(v, (0, 2, 1, 3))

    # QKT
    # output shape (b, nh, s, s)
    score = ops.bmm(q, k)

    # mask
    assert mask.shape == (b, s, s) 
    score = ops.where(mask.view(b, 1, s, s), score, -1e4) 
    score = ops.softmax(score)

    # (b, nh, s, s) * (b, nh, s, h1) -> (b, nh, s, h1)
    attnOut = ops.bmm(score, v)        
    
    # (b, nh, s, h1) -> (b, s, nh, h1) -> (b, s, h)
    attnOut = ops.permute(attnOut, (0, 2, 1, 3)).flatten(start_dim=2)
   

    assert attnOut.dtype == dtype.float32
    return attnOut 
    
def mha_decode(q:Tensor, k:Tensor, v:Tensor, mask:Tensor, numHead:int) :
    assert q.dtype == dtype.float32
    assert k.dtype == dtype.float32
    assert v.dtype == dtype.float32

    assert q.shape[1] == 1
    b, s, h = k.shape
    # s include the token generated in this iteration

    assert h % numHead == 0
    headDim = h // numHead

    scaling = headDim ** -0.5

    # (b, 1, nh, h1)
    q = q.view(b, 1, numHead, headDim) * scaling
    k = k.view(b, s, numHead, headDim)
    v = v.view(b, s, numHead, headDim)

    # (b, 1, nh, h1) -> (b, nh, 1, h1)/(b, nh, h1, s)
    q = ops.permute(q, (0, 2, 1, 3))
    k = ops.permute(k, (0, 2, 3, 1))
    v = ops.permute(v, (0, 2, 1, 3))

    # output shape (b, nh, s, s)
    score = ops.bmm(q, k)

    # mask
    assert mask.shape == (b, s)
    score = ops.where(mask.view(b, 1, 1, s), score, -1e4)
    score = ops.softmax(score)

    # (b, nh, 1, s) * (b, nh, s, h1) -> (b, nh, 1, h1)
    attnOut = ops.bmm(score, v)        
    
    # (b, nh, 1, h1) -> (b, 1, nh, h1) -> (b, 1, h)
    attnOut = ops.permute(attnOut, (0, 2, 1, 3)).flatten(start_dim=2)
    
    return attnOut