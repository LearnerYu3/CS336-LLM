import torch

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.causal_mha import CausalMultiHeadAttention
from cs336_basics.swiglu import SwiGLU

class TransformerBlock(torch.nn.Module):
    """
    Pre-norm Transformer block with optional RoPE.
    x -> pre-norm -> CausalMHA -> +x -> pre-norm -> FFN -> +x
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        use_rope: bool = True,
        device: torch.device|None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.use_rope = use_rope

        # CausalMHA
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.attn = CausalMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        # FFN
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        token_positions = None

        if self.use_rope:
            *batch_size, seq_len, _ = x.shape

            # 内部生成 token_positions 
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            token_positions = token_positions.view(
                *([1] * len(batch_size)), seq_len
            ).expand(*batch_size, seq_len)

        # x += self.attn(...) 会 in-place 修改，造成计算图错误，计算梯度时会出错，训练时避免使用 +=
        x = x + self.attn(self.ln1(x), token_positions=token_positions)

        x = x + self.ffn(self.ln2(x))

        return x
