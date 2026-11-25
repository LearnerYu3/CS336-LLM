import torch

from cs336_basics.linear_module import Linear
from cs336_basics.rope import RotaryPositionalEmbedding

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax
    Args:
    x: shape (..., d_k)
    dim: dimension to apply softmax
    Returns:
    Float[Tensor, " ... sequence_length d_k"]: Tensor with softmax applied
    """

    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max

    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor|None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention with optional causal masking.
    Args:
    Q: shape (batch_size, ..., sequence_length, d_k)
    K: shape (batch_size, ..., sequence_length, d_k)
    V: shape (batch_size, ..., sequence_length, d_k)
    is_causal: whether to apply causal masking
    Returns:
    Float[Tensor, " ... sequence_length d_k"]: Tensor with scaled dot-product attention
    """
    d_k = K.shape[-1]
    scale = 1.0 / (d_k ** 0.5)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weight = softmax(scores, dim=-1)

    output = torch.matmul(attn_weight, V)
    return output

class CausalMultiHeadAttention(torch.nn.Module):
    """
    Causal multi-head self-attention with rope.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device: torch.device|None = None,
        dtype: torch.dtype|None = None,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.max_seq_len = max_seq_len

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE 作用在 K 和 V 上
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len,
            device=device,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor|None = None) -> torch.Tensor:
        """
        Forward pass of the causal multi-head self-attention.
        Args:
        x: shape (batch_size, ..., sequence_length, d_model)
        Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with causal multi-head self-attention
        """
        *batch_size, seq_len, _ = x.shape

        # 内部生成 token_positions 
        # token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        # token_positions = token_positions.view(
        #     *([1] * len(batch_size)), seq_len
        # ).expand(*batch_size, seq_len)

        # 投影 Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape -> (batch_size, ..., num_heads, seq_len, d_k)
        q = q.view(*batch_size, seq_len, self.num_heads, self.d_k)
        k = k.view(*batch_size, seq_len, self.num_heads, self.d_k)
        v = v.view(*batch_size, seq_len, self.num_heads, self.d_k)

        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        if token_positions is not None:
            # RoPE on Q and K only
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # causal masking: True = keep (lower triangle incl. diagonal)
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=0,
        )

        attn_output = scaled_dot_product_attention(q, k, v, mask=mask)

        # reshape -> (batch_size, ..., sequence_length, d_model)
        attn_output = attn_output.transpose(-3, -2)
        attn_output = attn_output.contiguous().view(
            *batch_size, seq_len, self.d_model
        )

        return self.output_proj(attn_output)
