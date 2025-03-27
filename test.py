import torch
import torch.nn.functional as F
from torch import Tensor
import math
from torchBackend import mha_prefill


def mha_prefill_ref(q: Tensor, k: Tensor, v: Tensor, mask: Tensor, numHead: int):
    """
    Multi-head attention prefill implementation.
    Args:
        q: query tensor of shape (batch_size, seq_len, embed_dim)
        k: key tensor of shape (batch_size, seq_len, embed_dim)
        v: value tensor of shape (batch_size, seq_len, embed_dim)
        mask: attention mask tensor of shape (batch_size, seq_len)
        numHead: number of attention heads
    Returns:
        attention output tensor of shape (batch_size, seq_len, embed_dim)
    """
    batch_size, seq_len, embed_dim = q.shape
    head_dim = embed_dim // numHead
    
    # Split into multiple heads
    q = q.view(batch_size, seq_len, numHead, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, numHead, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, numHead, head_dim).transpose(1, 2)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # Apply mask
    mask = mask.unsqueeze(1).unsqueeze(1)  # shape: (batch_size, 1, 1, seq_len)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Compute attention weights
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, v)
    
    # Concatenate heads
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    
    return output

def test_mha_prefill():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test configurations
    test_cases = [
        # (batch_size, seq_len, embed_dim, num_heads)
        (1, 10, 64, 4),
        (2, 5, 128, 8),
        (4, 1, 32, 2),
        (1, 100, 256, 16),
    ]
    
    for batch_size, seq_len, embed_dim, num_heads in test_cases:
        print(f"\nTesting with batch_size={batch_size}, seq_len={seq_len}, "
              f"embed_dim={embed_dim}, num_heads={num_heads}")
        
        # Create random tensors
        q = torch.randn(batch_size, seq_len, embed_dim)
        k = torch.randn(batch_size, seq_len, embed_dim)
        v = torch.randn(batch_size, seq_len, embed_dim)
        
        # Create mask (randomly mask some positions)
        mask = torch.ones(batch_size, seq_len)
        mask[:, :seq_len//2] = 0  # Mask first half of sequence
        
        # Run MHA
        output = mha_prefill(q, k, v, mask, num_heads)
        ref = mha_prefill_ref(q, k, v, mask, num_heads)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, embed_dim), \
            f"Output shape {output.shape} doesn't match expected {(batch_size, seq_len, embed_dim)}"
        
        # Check masked positions don't contribute to output
        # For masked positions, output should be close to zero
        masked_output = output[:, :seq_len//2, :]
        assert torch.allclose(masked_output, torch.zeros_like(masked_output), atol=1e-6), \
            "Masked positions should have near-zero output"
        
        print("Test passed!")
    
    # Test edge cases
    print("\nTesting edge cases...")
    
    # Empty sequence
    q = torch.randn(1, 0, 64)
    k = torch.randn(1, 0, 64)
    v = torch.randn(1, 0, 64)
    mask = torch.ones(1, 0)
    output = mha_prefill(q, k, v, mask, 4)
    assert output.shape == (1, 0, 64), "Empty sequence test failed"
    print("Empty sequence test passed!")
    
    # Single element sequence
    q = torch.randn(1, 1, 32)
    k = torch.randn(1, 1, 32)
    v = torch.randn(1, 1, 32)
    mask = torch.ones(1, 1)
    output = mha_prefill(q, k, v, mask, 2)
    assert output.shape == (1, 1, 32), "Single element test failed"
    print("Single element test passed!")
    
    # All positions masked
    q = torch.randn(2, 5, 128)
    k = torch.randn(2, 5, 128)
    v = torch.randn(2, 5, 128)
    mask = torch.zeros(2, 5)
    output = mha_prefill(q, k, v, mask, 8)
    assert torch.allclose(output, torch.zeros_like(output), "All masked test failed")
    print("All masked test passed!")
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_mha_prefill()