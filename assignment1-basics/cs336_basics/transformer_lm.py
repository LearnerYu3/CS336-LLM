import torch

from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.embedding import Embedding
from cs336_basics.linear_module import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.causal_mha import softmax


class TransformerLM(torch.nn.Module):
    """
    Decoder-Only Transformer Language Model
    token_ids (B, T)
     -> token_embedding
     -> N x TransformerBlock (pre-LN, CausalMHA + RoPE, pre-LN, SwiGLU)
     -> final RMSNorm
     -> Linear (d_model -> V)
     -> logits (B, T, V)  # unnormalized
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device|None = None,
        dtype: torch.dtype = torch.float32,
        weight_tying: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
                use_rope=True,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype,
        )

        if weight_tying:
            # 共享权重
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context length {self.context_length}")

        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits