import torch
import math
from torch import nn
from einops import einsum, rearrange

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
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=init_std, a=-3 * init_std, b=3 * init_std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... in_features, out_features in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device|None = None, dtype: torch.dtype|None = None):
        """
        Construct an embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e., d_model
            device (torch.device|None): Device to store the parameters on
            dtype (torch.dtype|None): Data type of the parameters
        """
        super().__init__()
        self.weights = nn.Parameter(torch.zeros((num_embeddings, embedding_dim), dtype=dtype, device=device))
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=1, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up the embedding vectors for the given IDs.

        Args:
            token_ids (torch.Tensor): The batch of input token id sequences, a tensor of shape (batch_size, sequence_length)
        """
        return self.weights[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device|None = None, dtype: torch.dtype|None = None):
        """
        Constructs an RMSNorm module which computes the Root Mean Square normalization based on https://arxiv.org/abs/1910.07467.
        RMSNorm(a[i]) := (a[i] / RMS(a)) * g[i]

        Where: 
        RMS(a) := sqrt((1/d_model) * sum(i in 0..d_model: a[i]**2 + eps) )

        d_model (int): Hidden dimension of the model.
        eps (float): Epsilon value for numeric stability
        device (torch.device | None): Device to store the parameters on
        dtype: (torch.dtype | None): Data type of the parameter
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
    
    def forward(self, x: torch.tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32) # Upcast to avoid overflow when computing squares
        # x is (batch_size, d_model)
        assert x.shape[-1] == self.d_model
        rms = torch.sqrt((einsum(x ** 2 + self.eps, "... d_model -> ...") / self.d_model)) # rms -> (batch_size,)
        rms = rearrange(rms, "... -> ... 1") # rearrange to (batch_size, 1) so broadcasting works when dividing (batch_size, d_model) below
        result = (x / rms) * self.g # -> (batch_size, d_model)
        # restore original data type
        return result.to(in_dtype)

class FFSwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int|None = None, device: torch.device = None, dtype: torch.dtype|None = None):
        """
        Constructs a position-wise feed-forward network with SwiGLU activation:

        FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x), 

        d_model (int): Dimensionality dimension of the model.
        d_ff (int): Dimentionality of the feedfoward network
        device (torch.device | None): Device to store the parameters on
        dtype: (torch.dtype | None): Data type of the parameter
        """

        super().__init__()
        # Dimensionality of feedfoward network, make sure it's a multiple of 64
        self.d_ff = d_ff if d_ff is not None else math.ceil((d_model * 8/3) / 64) * 64
        self.W1 = Linear(in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(in_features=self.d_ff, out_features=d_model, device=device, dtype=dtype)
        self.W3 = Linear(in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype)
    
    def forward(self, x:torch.tensor):
        # SiLU = x * sigmoid(x)
        w1_x = self.W1(x)
        w3_x = self.W3(x)
        y = w1_x * torch.sigmoid(w1_x)
        y = y * w3_x
        y = self.W2(y)
        return y
