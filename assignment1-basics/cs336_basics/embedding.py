import torch
import math

class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device|None=None,
        dtype: torch.dtype|None=None
    ):
        """
        Embedding layer.
        Args:
            num_embeddings: The number of embeddings.
            embedding_dim: The dimension of the embeddings.
            device: The device to use.
            dtype: The dtype to use.
        """
        super().__init__()
        self.d_model = embedding_dim
        self.vocab_size = num_embeddings

        self.W = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize the weight matrix drawing from N(0, 2/(in+out)), truncated normal distribution
        std = 1.0
        torch.nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer.
        Args:
            token_ids: The token ids.
        Returns:
            The embeddings.
        """
        # self.W[token_ids] 不支持padding，需要手动处理padding
        # F.embedding(token_ids, self.W, padding_idx=padding_idx) 处理会更好
        return self.W[token_ids]