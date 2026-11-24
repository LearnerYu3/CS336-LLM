import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device|None = None,
    ):
        """
        Rotary Positional Embedding
        
        Args:
            theta (float): The angle of the rotary encoding.
            d_k (int): The dimension of the key.
            max_seq_len (int): The maximum sequence length.
            device (torch.device|None, optional): The device to use. Defaults to None.
        """
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 1) inv_freq: (d_k/2, ) fp32
        pair_idx = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = theta ** (-pair_idx / d_k)

        # 2) position: (max_seq_len, )
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # 3) angles: (max_seq_len, d_k/2)
        angles = torch.outer(positions, inv_freq)

        # 4) cache cos/sin, persitent = False
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the rotary position embedding.
        
        Args:
            x (torch.Tensor): The input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): The token positions of shape (..., seq_len,).
        Returns:
            torch.Tensor: The output tensor of shape (..., seq_len, d_k).
        """
        origin_dtype = x.dtype
        x = x.to(torch.float32)

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.stack((out_even, out_odd), dim=-1)
        out = out.flatten(-2)

        return out.to(origin_dtype)
        