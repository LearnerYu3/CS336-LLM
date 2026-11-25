import torch
import math


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device|None = None,
        dtype: torch.dtype|None = None,
    ):
        """
        RMSNorm layer.
        Args:
            d_model: The dimension of the input.
            eps: The epsilon value to use in the RMSNorm.
            device: The device to use.
            dtype: The dtype to use.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RMSNorm layer.
        Args:
            x: The input.
        Returns:
            The output.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # keepdim 保留最后一维不被压缩，保持与输入相同的形状，避免广播错误
        rms = torch.sqrt(
            torch.mean(x**2, dim=-1, keepdim=True) + self.eps
        )
        x_normed = x / rms

        return (x_normed * self.weight).to(in_dtype)
