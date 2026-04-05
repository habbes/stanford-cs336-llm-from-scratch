import torch
import math
from torch import nn
from einops import einsum, rearrange, repeat

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

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device=None):
        """
        Constructs the RoPE module and create buffers if needed. This module
        can be reused to perform rotations for all query or key vectors in any layers.

        Args:
            theta (float): The constant used in the denominator of the RoPE equation.
            d_k (int): The dimension of query and key vectors
            max_seq_len (int): Maximum sequence length of the input.
            device (torch.device): Device to store the buffer on.
        """
        super().__init__()
        
        # RoPE treats input query and key vectors as collection of
        # coordinate pairs in 2D space then rotates those points
        # using a 2x2 rotation matrix.
        # The angle of the rotation is based on the token position
        # and the dimension of the query/key vectors.

        # We create and cache a tensor of 2D rotation blocks for all possible
        # token positions up to max_seq_len

        # First let's compute the angles for all possible positions.
        # The rotation angle[pos, k] = i/(theta ** (2k - 2)/d)
        # where k is in range [1, d/2] and pos is the token position
        # which means (2k-2) goes from 0, 2, 6, ..., d-2
        assert d_k % 2 == 0, f"The d_k parameter representing query vector dimensions should be an even number, but got {d_k}"
        positions = rearrange(torch.arange(max_seq_len, device=device), "... -> ... 1")
        k = torch.arange(1, d_k / 2 + 1, device=device)
        denom = theta ** ((2 * k - 2) / d_k)

        assert denom.shape == (d_k / 2,)
        angles = positions / denom

        # Each rotation matrix R[pos, k] = 
        # [ 
        # cos(angle[pos, k]),  -sin(angle[pos, k])
        # sin(angle[pos, k]),   cos(angle[pos, k])
        #]
        # So we need to compute cos and sin for all possible
        # angles.
        cosines = torch.cos(angles)
        sines = torch.sin(angles)

        assert cosines.shape == (max_seq_len, d_k / 2)
        assert sines.shape == (max_seq_len, d_k / 2)

        # For each token position, we can pack the 2x2 rotation
        # matrices into a larger matrix R[pos] so we can
        # efficiently rotate different points in the input vector
        # in parallel.
        # The full matrix R[pos] will is a block digonal matrix
        # R[pos] = 
        # [
        #   R[pos, 1]    0            0              0
        #   0            R[pos, 2]    0              0
        #   0            0            R[pos, 3]  ... 0
        #   0            0            0          ... R[pos, d/2]
        # ]
        # Where each R[pos, k] is a 2x2 rotation matrix
        # and each 0 is a 2x2 zero matrix.
        # So expanded, it looks like (cos and sin arguments omitted for brevity)
        # [[cos, -sin,    0,      0,      0,         0]
        #  [sin,  cos,    0,      0,      0,         0]
        #  [0,    0,      cos,    -sin,   0,         0]
        #  [0,    0,      sin,    cos,    0,         0]
        #  [0,    0,      0,      0,      cos,      -sin]
        #  [0,    0,      0,      0,      sin,      cos]
        # ]
        # From this we can see that for each 2x2 rotation sub-matrix:
        # - cosines at the top-left of the sub-matrix are at indices [0,0], [2,2], [4,4] etc. i.e. [even_idx, even_idx]
        # - -sines at the top-right are at indices [0,1], [2,3], [4,5], etc. i.e. [even_idx, odd_idx]
        # - sines at the bottom-left are at indices [1,0], [3,2], [5,4], etc., i.e. [odd_idx, even_idx]
        # - cosines at the bottom-right are at indices [1,1], [3,3], [5,5],e etc., i.e. [odd_idx, odd_idx]

        even_idx = torch.arange(0, d_k, 2, device=device, dtype=torch.int)
        odd_idx = torch.arange(1, d_k, 2, device=device, dtype=torch.int)

        assert even_idx.shape == (d_k / 2,)
        assert odd_idx.shape == (d_k / 2,)

        rotation_matrix = torch.zeros((max_seq_len, d_k, d_k), device=device)
        rotation_matrix[:, even_idx, even_idx] = cosines
        rotation_matrix[:, even_idx, odd_idx] = -sines
        rotation_matrix[:, odd_idx, even_idx] = sines
        rotation_matrix[:, odd_idx, odd_idx] = cosines

        # Register the matrix so the Module is aware of it.
        # This way if this module is moved to a different device, the matrix
        # will be moved to. However since it has no learnable parameters,
        # we don't use nn.Parameter.
        # We also don't want this to be (de)serialized when the model is
        # saved or loaded, hence persistent=False.
        self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies RoPE to the specified input x (usually a matrix of query or key vectors for each
        position in the sequence) for the specified positions.

        Processes an input tensor of shape (..., seq_len, d_k) and returns a tensor
        of the same shape.

        Args:
            x (torch.Tensor): The input, usually key or query, to apply RoPE to. Shape (..., seq_len, d_k)
            token_positions (torch.Tensor): Token positions for which to apply RoPE. Shape = (..., seq_len)
        """
        # Retrieve rotation matrices for the specified token positions
        rm = self.rotation_matrix[token_positions]
        # For each input vector in the batch x in each token position, we want to perform
        # matrix multiplication R[pos] @ x[batch, pos].T
        # that's equivalent to x[batch, pos] @ R[pos].T
        result = einsum(x, rm, "... n_positions in_features, n_positions out_features in_features -> ... n_positions out_features ")
        return result