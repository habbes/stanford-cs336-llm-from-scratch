import torch
import math
from torch import nn
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float, Int

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the softmax exp(x)/sum(exp(x)) which
    transforms a tensor of unnormalized values into
    a probability distributions, i.e. values ranging from 0 to 1.

    Args:
        x (torch.Tensor): The input tensor
        dim (int): The dimension along which to compute the softmax.
    """
    # exp(x) can be unboundendly large, and could lead to inf/inf which results to NaN
    # To avoid this overflow, we can subtract the max value from each value
    # such that the max becomes x is 0. This prevents overflow.
    # This works because exp(x + c)/sum(exp(x + c)) == exp(x)/sum(exp(x))
    maxes = torch.max(x, dim=dim, keepdim=True).values
    exps = torch.exp(x - maxes)
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)
    return exps / sum_exps

def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Performs scaled dot-product attention

    softmax(Q @ K.T / sqrt(d_k)) @ V

    Args:
        queries (torch.Tensor): Batch of query matrices Q. Shape = (batch_size, ..., n, d_k)
        keys    (torch.Tensor): Batch of key matrices K. Shape = (batch_size, ..., m, d_k)
        values  (torch.Tensor): Batch of value matrics V. Shape = (batch_size, ..., m, d_v)
        mask    (torch.Tensor|None): Optional boolean mask to prevent specific queries from attending to specific keys. Shape = (n, m)
            If mask[i, j] is True, then query i should attend to key j, otherwise it should not
            (i.e. the weight associated with query and k should be 0).
    
    Where n = length of source sequence, m = length of target sequence
    
    Returns:
        Tensor of shape (batch_size, ..., n, d_v)
    """
    # In each batch item, there are n query vectors of size d_k and m key vectors for size d_k
    # where n = source sequence length, m = target sequence length
    # Since this is a generalized utility function, we don't force the assumption that n == m.

    # Computes Q @ K.T
    dot_product = einsum(queries, keys, "batch_size ... n d_k, batch_size ... m d_k -> batch_size ... n m")
    scaled_dot_product = dot_product / math.sqrt(queries.shape[-1])

    # For each item in the batch, the dot product computes a matrix Q @ K.T of shape (n, m)
    # where each element i_j shows how strongly query i relates to key j.
    # So each row i shows how strongy query i relates to each key in the target sequence.
    if mask is not None:
        # Set locations we want to mask out to -inf
        # Where mask is false, the the corresonding query key should not attend to the key, so we want
        # the dot product value at that position to be 0, and we do that by setting each position
        # where mas is false to -inf
        # because exp(-inf) == 0, so the corresponding query, key and value will not contribute to the attention.
        scaled_dot_product = scaled_dot_product.masked_fill(~mask, -torch.inf)

    # We use softmax to normalize the weights into probability distributions that sum up to 1
    # for each row each in the n*m matrix. i.e. for each row in the n*m matrix, the column values
    # in that row should sum up to 1.
    # Since we're applying softmax against the m columns, that's the last dimension
    # hence setting dim to -1
    weights = softmax(scaled_dot_product, dim=-1)

    # In each batch multiply the n*m weights matrix by the m*d_v values matrix (m value vectors of size d_v)
    # This results in an n*d_v output matrix, where each row i is a merged/combined value vector that's a 
    # weighted average of all the value vectors based on the weights corresponding to query i.
    # So each row is a different combination of the value vectors based on the weights corresponding
    # to that row, and these weights represents how strongly the position corresponding to that row
    # attends to each target position's key and value.
    # And by weighted average, we mean that each component j of the output row i is a weighted average
    # of the kth component of each the m value vectors, based on the m weights in row i
    # so the stronger the query i attends to key j, the more vector j's d_v components will contribute
    # to the output vector
    result = einsum(weights, values, "batch_size ... n m, batch_size ... m d_v -> batch_size ... n d_v")
    return result


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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, **kwargs):
        """
        Constructs the Multi-Head Causal Self-Attention module.

        Args:
            d_model: size of the input embedding dimension
            num_heads: number of heads
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        d_k = d_model // num_heads
        d_v = d_k
        self.Wq = Linear(d_model, num_heads * d_k)
        self.Wk = Linear(d_model, num_heads * d_k)
        self.Wv = Linear(d_model, num_heads * d_v)
        self.Wo = Linear(num_heads * d_v, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies MultiHeadSelfAttention to the sequence:

        y = MultiHead(x @ Wq, x @ Wk, x @ Wv) @ Wo

        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)
            where head_i = Attention(Q_i, K_i, V_i)

        Args:
            x: batched input sequence, shape (b, n, d_model)
        
        Returns:
            y: attention-aware sequence, shape (b, n, d_model)
        """
        # Use the weight parameters Wq, Wk, and Wv
        # to compute the different heads of queries, keys and values
        # We'll have a query vector per head per token position in batch item
        # Assuming x -> (b, n, d_model)
        # TODO: consider using a single weight matrix so you only need
        # one matrix multiplication to compute Q, K and V.
        Q = self.Wq(x) # Q -> (b, n, h * d_k)
        K = self.Wk(x) # K -> (b, n, h * d_k)
        V = self.Wv(x) # V -> (b, n, h * d_v)

        # Note, we should see the h * d_k dimension in Q as h independent query vectors of size d_k

        # For each input sequence in the batch, we want to compute attention
        # independently in each head. So we compute h separate attention operations
        # in parallel per batch item. We don't multiply an entire row of Q with
        # an entire column of K.T, instead we multiply a row in one head of Q
        # with a column in the corresponding head in K.T. We also perform masking
        # and softmax independently per head, and so on and so forth.
        # For this reason, we should treat h as another batch dimension, i.e.
        # a sub-array that contains h (n, d_k) matrices.
        # This will make masking and scale_dot_product_attention functions convenient
        # since they're already design to operate on the last 2 dimensions and broadcast
        # across batches.
        query_heads = rearrange(Q, "... n (h d_k) -> ... h n d_k", h=self.num_heads)
        key_heads = rearrange(K, "... n (h d_k) -> ... h n d_k", h=self.num_heads)
        value_heads = rearrange(V, "... n (h d_v) -> ... h n d_v", h=self.num_heads)

        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))

        multi_head_attention = scaled_dot_product_attention(
            queries=query_heads,
            keys=key_heads,
            values=value_heads,
            mask=causal_mask
        ) # -> (b, h, n, d_v)

        # After we've computed multi head attention, which results in
        # h independent heads of attention-aware (n, d_v) matrices,
        # we want to unwrap the heads into a single matrix of shape (n, h * d_v).
        # This allows us to apply the transformation Wo to the matrices.
        # We can think of Wo as a fusion of the different heads, that learns
        # to mix the information indepdently learned by different heads into
        # a merged representation in the d_model embedding space
        o = rearrange(multi_head_attention, "... h n d_v -> ... n (h d_v)")
        y = self.Wo(o) # y -> (b, n, d_model)
        return y
        
