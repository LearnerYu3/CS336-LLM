from torch import nn
import torch
import math
from einops import einsum

class Linear(nn.Module):
    """
    A custom linear layer without bias, y = x W^T internally,
    but weight is stored as W of shape (out_features, in_features).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device|None=None,
        dtype: torch.dtype|None=None,
    ):
        """
        Initialize a linear layer with the given input and output features.
        Args:
            in_features: The dimension of input.
            out_features: The dimension of output.
            device: The device to store the parameters.
            dtype: Data type of parameters.
        """
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize the weight matrix drawing from N(0, 2/(in+out)), truncated normal distribution
        std = math.sqrt(2.0 / (self.d_in + self.d_out))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear layer.
        Args:
            x: The input tensor.
        Returns:
            The output tensor.
        """
        # return einsum(x, self.W, "... in, out in -> ... out")
        # torch.einsum 的方程里每个维度必须是单个字母，"in" 被解析为 'i' + 'n' = 两个字母，可以使用 einops.einsum 代替
        return torch.einsum("... i, o i -> ... o", x, self.weight)
