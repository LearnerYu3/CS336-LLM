import math
import torch

from cs336_basics.linear_module import Linear

class SwiGLU(torch.nn.Module):
    """
    SwiGLU Feed-Forward Network.
    y = (SiLU(x @ W1.T) * (x @ W3.T)) @ W2.T
    where SiLU(x) = x * sigmoid(x), aka Swish
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int|None = None,
        device: torch.device|None = None,
        dtype: torch.dtype|None = None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = int(math.ceil((8/3) * d_model / 64) * 64)

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype) # gate projection
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype) # output projection
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype) # value projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        return: (batch_size, seq_len, d_model)
        """

        gate = self.w1(x)
        gate = gate * torch.sigmoid(gate)
        value = self.w3(x)

        return self.w2(gate * value)