import torch
import math
from torch import nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device:torch.device = None, dtype: torch.dtype = None):
        """
        Constructs a linear transformation module.

        Args:
            in_features (int): final dimension of the input
            out_features (int): final_dimension of the output
            device (torch.device | None): device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        self.weights = nn.Parameter(torch.zeros((out_features, in_features), dtype=dtype, device=device))
        # Initialize from N(mean = 0, variance = 2 / (d_in + d_out) truncated at [-3stddev, 3stddev]
        init_std = math.sqrt(2 / (in_features + out_features))
        # Note that initialization functions are execute in no_grad mode so that they don't
        # affect the automatic gradient computation graph
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=init_std, a=-3 * init_std, b=init_std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... in_features, out_features in_features -> ... out_features")
