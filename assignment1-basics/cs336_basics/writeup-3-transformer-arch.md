# 3. Transformed Language Model Architecture

Related papers:

- [Improving Language Understanding by Generative Pretraing, Radford et al 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

This is the architecture we're going to implement (this differs from the original Transformer paper. Various sections will explain how and why we deviate from the original)

![Our transformer architecture](our-transformer-architecture.png)


## 3.3 Remark: Batching, Einsum and Efficient Computation

Einsum notation, inspired by Einstein sums, is a more readable and ergonomic way of working tensor dimensions.
It provides more intuitive syntax of operations like rearranging dimensions, transposition, broadcast operations, etc.

The instructions recommend that students new to einsum notation use the [**`einops`**](https://einops.rocks/) package and that those
that are familiar to use the [**`einx`**](https://einx.readthedocs.io/en/latest/). Since
this is my first time hearing of this, I'm going to use **einops**:

- [einops tutorials](https://einops.rocks/1-einops-basics/)
- [einops repository](https://github.com/arogozhnikov/einops)
- [einops in 30 seconds](https://medium.com/@kyeg/einops-in-30-seconds-377a5f4d641a)

I've created a repo to learn and experiment with einops: https://github.com/habbes/learn-einops

Reiterating the instructions in the assignment overview, this is the policy of the course on what abstractions
can be used. I want to abide by it as well:

> We expect you to build these components from scratch. In particular, you may notuse any definitions from `torch.nn`, `torch.nn.functional`, or `torch.optim` except for the following:
>
> - `torch.nn.Parameter`
> - [Container classes](https://pytorch.org/docs/stable/nn.html#containers) in `torch.nn` (e.g., `Module`, `ModuleList`, `Sequential`, etc.)
> - The `torch.optim.Optimizer` base class
>
> You may use any other PyTorch definitions. When in doubt, consider if using it compromises the“from-scratch” ethos of the assignment

Speaking of PyTorch, I also needed a refresher. I took a break to go through some tutorials here: https://github.com/habbes/learn-pytorch

### Notes on mathematical notation memory-ordering

> Many machine learning papers use row vectors in their notation, which result in representations that mesh well with the row-major memory ordering used by default in NumPy and PyTorch. 
> With row vectors, a linear transformation looks like `y=xW.⊤`,
> for row-major W of shape `(d_out, d_in)` and row-vector x of shape `(1, d_in)`.
>
> In linear algebra it’s generally more common to use column vectors, where linear transformations look like `y=W x`,
> given a row-major W  of shape `(d_out, d_in)` and column-vector x of shape `(d_in,)`.
> We will use **column vectors** for mathematical notation in this assignment, as it is generally easier to follow the math this way.
> You should keep in mind that if you want to use plain matrix multiplication notation, you will have to apply matrices using the row vector convention, since PyTorch uses row-major memory ordering.
> If you use einsum for your matrix operations, this should be a non-issue.

*Note*: I'm using a shape to define the matrix set for convenience, instead of the standard mathematical notation of R with the shape as subscript.

To make this more concrete, let's go through an example.

Let's say we have a 3x2 matrix `w` representing the weights of the linear transformation:

```
[
   2 1
   4 3
   5 0
]
```

And 1x2 vector `x` representing the input object. We can consider this as a single object with 2 features:

```
[5 2]
```

Let's encode this in PyTorch with `W` as tensor with shape (3,2) and `x` as tensor with shape (1, 2):

```python
>>> W = torch.tensor([[2, 1], [4, 3], [5,0]])    
>>> W.shape
torch.Size([3, 2])
>>> W
tensor([[2, 1],
        [4, 3],
        [5, 0]])
>>> x = torch.tensor([[5, 2]])
>>> x.shape
torch.Size([1, 2])
>>> x
tensor([[5, 2]])
```

The 2 columns of the W matrix match the 2 features of the input object vectors. And the 3 rows of the matrix W
mean we desire 3 output features per output object. That's why we can think of the dimensions as x -> (1, d_in)
and W -> (d_out, d_in)

If we want to compute the linear transformation y = Wx, then we should treat x as a column vector with shape (2, 1)
so that the dimensions align properly. In PyTorch, that would be `W @ x.T`

```python
>>> x.T   
tensor([[5],
        [2]])
>>> W @ x.T
tensor([[12],
        [26],
        [25]])
```

The result is the 3x1 column vector:

```
[
  12
  26
  25
]
```

However, sticking to the original row vector, and row-major memory ordering, we can perform the "equivalent"
transformation using `x @ W.T` where we transpose W into a 2x3 matrix, but keep x as a 1x2 row vector:

```python
W.T
tensor([[2, 4, 5],
        [1, 3, 0]])
>>> x @ W.T
tensor([[12, 26, 25]])
```

The result is the 1x3 row vector:

[
   12 26 25
]

Note that this is the same result as the previous one, but transposed, i.e. `x @ W.T == (W @ x.T).T`

We can achieve the row-major result using `einsum` without explicit transpose:

```python
>>> einsum(x, W, "batch_size d_in, d_out d_in -> batch_size d_out")
tensor([[12, 26, 25]])
```

We also achieve them same result if we swap the order of the tensors. It's cool
that this just works with `einsum` without explict transposes even though matmul
is not commutative (not sure if there's any difference in the low-level order of operations or efficiency):

```python
>>> einsum(W,x, "d_out d_in, num_objects d_in -> num_objects d_out")            
tensor([[12, 26, 25]])
```

If we wanted to get a column vector as result, we can just swap the output dimensions of the `einsum`:

```python
einsum(x, W, "batch_size d_in, d_out d_in -> d_out batch_size")
tensor([[12],
        [26],
        [25]])
```

We'll use the row major approach in computations as it's more common in deep learning. But more importantly,
it keeps the batch size (and other dimensions that should not be summed over) in the leading dimensions
with preserves batch semantics, and most pytorch nn operations expect the batch size as the leading dimension.
The row major form is also aligned with PyTorch's default memory order, deep learning kernels, etc. SIMD/CUDA, etc.

## 3.4 Basic Building Blocks: Linear and Embedding modules

### 3.4.1 Parameter initialization

> Training neural networks effectively often requires careful initialization of the model parameters—bad initial-izations can lead to undesirable behavior such as vanishing or exploding gradients.
> Pre-norm transformers are unusually robust to initializations, but they can still have a siginificant impact on training speed and convergence.
> Since this assignment is already long, we will save the details for assignment 3, and instead give you some approximate initializations that should work well for most cases. For now, use:

- Linear Weights: `NormalDistribution(mean = 0, variance = 2 / (d_in + d_out))` truncated at [-3stddev, 3stddev]
- Embedding: `NormalDistribution(mean = 0, variance = 1) truncated at [-3, 3]
- RMSNorm: 1

You should use `torch.nn.init.trunc_normal_` to initialize the truncated normal weights

### 3.4.2 Linear Module

In this part, I implement a linear module based on `nn.Module` to compute y = Wx, to mirror the standard [`nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.linear.Linear.html).

*Note*: Note that we do not include a bias term (i.e. b is conceptually 0 in y = Wx + b), following most modern LLMs.

> Make sure to:
>
> - subclass `nn.Module`
> - call the superclass constructor
> - construct and store your parameter as W (not W.T) for memory ordering reasons, putting it in an `nn.Parameter`
> - of course,don’t use `nn.Linear` or `nn.functional.linear`

Storing our weights as `W` instead of `W.T` essentially means using a tensor of shape `(d_out, d_in)` to store the matrix.

I've implemented the custom `Linear` module in the [`nn_modules.py`](./nn_modules.py) file.

To test this, first I update the `run_linear` function in [`../tests/adapters.py`](../tests/adapters.py) to call
initialize and call my custom `Linear` class.

Then run test as:

```sh
uv run pytest -k test_linear
```

```sh
uv run pytest -k test_linear     
======================================================================== test session starts ========================================================================
platform darwin -- Python 3.11.12, pytest-8.4.1, pluggy-1.6.0
rootdir: /Users/habbes/code/learn/stanford-cs336-llm-from-scratch/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                     

tests/test_model.py::test_linear PASSED

================================================================= 1 passed, 47 deselected in 0.84s ==================================================================
```

Got an error running the test on Windows though:

```sh
====================================================================================================================== ERRORS ======================================================================================================================
___________________________________________________________________________________________ ERROR collecting assignment1-basics/tests/test_tokenizer.py ____________________________________________________________________________________________ 
ImportError while importing test module 'C:\Users\clhabins\source\repos\learn\stanford-cs336-llm-from-scratch\assignment1-basics\tests\test_tokenizer.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\..\..\AppData\Roaming\uv\python\cpython-3.11.13-windows-x86_64-none\Lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
assignment1-basics\tests\test_tokenizer.py:5: in <module>
    import resource
E   ModuleNotFoundError: No module named 'resource'
================================================================================================================= warnings summary ================================================================================================================= 
assignment1-basics\tests\adapters.py:294
  C:\Users\clhabins\source\repos\learn\stanford-cs336-llm-from-scratch\assignment1-basics\tests\adapters.py:294: DeprecationWarning: invalid escape sequence '\T'
    """Given the weights of a Transformer language model and input indices,

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================================================================================================= short test summary info ============================================================================================================== 
ERROR assignment1-basics/tests/test_tokenizer.py
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

Note that while the test passes (at least on *nix systems), it loads weights provided by the tests and not the model's
randomly initialized weights. So it doesn't test whether initialization logic is correctly implemented. I did stumble
on a bug in my initialization code when randomly looking at the code, so I want to write a test for that as well.

I'll basically create a new instance of the module and check that the weights are within the desired bounds and that
the mean and std/variance are close enough to what we expect, i.e.:

```python
>>> d_in, d_out = 1000, 500
>>> l = Linear(d_in, d_out)
>>> target_std = math.sqrt(2/(d_in + d_out))
>>> target_std
0.03651483716701107
>>> l.weights.std()
tensor(0.0360, grad_fn=<StdBackward0>)
>>> l.weights.mean()
tensor(-8.9683e-06, grad_fn=<MeanBackward0>)
>>> target_min, target_max = -3 * target_std, 3 * target_std
>>> target_min, target_max
(-0.10954451150103323, 0.10954451150103323)
>>> l.weights.min(), l.weights.max()
(tensor(-0.1095, grad_fn=<MinBackward1>), tensor(0.1095, grad_fn=<MaxBackward1>))
```
The values seem to be within reasonable bounds of the expected constraints.

I've created a test for this in the [`playground.py`](./playground.py) file under the `test_linear_module_initialization()`
function:

```sh
uv run -m cs336_basics.playground
```

```sh
...
SCENARIO: Verify Linear module weights are initialized with expected distribution
Test passed!
```

### 3.4.3 Embedding module

The embedding layer maps integer token IDs into a vector space of dimension `d_model`, where `d_model` conceptually represents the number of features that represent a token i.e.
An embedding vector will be learned for each unique token id. The embedding module will hold these embedding vectors into a tensor of size `(vocab_size, d_model)` where
`vocab_size` is the number of unique token IDs in our vocabulary.

The input to the embedding module is a batched sequence of tokens, i.e. each item in a batch is a sequence of token ids, hence the input tensor has shape `(batch_size, sequence_length)`.
The sequence length is basically the context window, how many tokens do we have to look back to in order to predict the next token.

Given this input, the embedding module returns the corresponding embedding vector for each input token. I.e., for each input token in each token sequence in the batch, index into
the embedding matrix and return te corresponding embedding vector. Hence the output of the embedding module will be a tensor of shape `(batch_size, sequence_length, d_model)`.

For example, let's say we have a vocabulary of size 3 with token Ids: `vocab = [0, 1, 2]`. Let `d_model = 2`, and we have the following embedding matrix:

```python
[
   [0.1, 0.5],
   [0.24, -1.45],
   [0.51, -2.3]
]
```

This means we have the following token ID to embedding vector mapping:

- 0 -> [0.1, 0.5]
- 1 -> [0.24, -1.45]
- 2 -> [0.51, -2.3]

Now let's say we have the following batch of inputs sequences `x`, with shape `(batch_size = 2, sequence_length = 4)`:

```python
batch =
[
   [0, 2, 1, 1],
   [1, 0, 2, 1]
]
```

Then the output of the embedding layer will be the following tensor of shape `(batch_size = 2, sequence_length = 4, d_model = 2)`

```python
[
    [
        [0.1, 0.5], [0.51, -2.3], [0.1, 0.5], [0.1, 0.5]
    ],
    [
        [0.24, -1.45], [0.1, 0.5], [0.51, -2.3], [0.24, -1.45]
    ]
]
```

Conveniently, PyTorch allows you to index a tensor using a collection of indices to retreive query items at the same time.
Like in the following example we use the `indices` collection to query the `x` tensor, which returns an output tensor
based on the specified indices:

```python
>>> x = torch.tensor([1,2,3,4,5,6])
>>> indices = [0, 5, 5, 4]
>>> x[indices]
tensor([1, 6, 6, 5])
```

And "batch indices" just work out of the box thanks to PyTorch broadcasting. In the following example we a `double` tensor
which is a sequence of even numbers starting at 0. Then we we give at a `(2, 3)` batch of index sequences (batch size = 2, sequence length = 3).
This returns a `(2, 3)` tensor of doubles corresponding to the input indices:

```python
>>> doubles = torch.tensor([0, 2, 4, 6, 8, 10])       
>>> batched_indices = torch.tensor([[1, 4, 3], [2, 3, 5]])
>>> doubles[batched_indices]
tensor([[ 2,  8,  6],
        [ 4,  6, 10]])
```

Well will this also work if the tensor we're indexing into has multiple dimensions? Yes it does! Instead of the 1-dimensional `doubles` vector,
let's use a `doubles_and_triples` tensor where each element is a 2-item vector containing both the double and triple of the corresponding index.
So we have `batched_indices` of shape (2,3) and `doubles_and_triples` of shape `(10, 2)` and the result is a tensor of shape `(2, 3, 2)`:

```python
>>> doubles_and_triples = torch.tensor([[0, 0], [2, 3], [4, 6], [6, 9], [8, 12], [10, 15]])
>>> batched_indices = torch.tensor([[1, 4, 3], [2, 3, 5]])
>>> doubles_and_triples[batched_indices]
tensor([[[ 2,  3],
         [ 8, 12],
         [ 6,  9]],

        [[ 4,  6],
         [ 6,  9],
         [10, 15]]])
```

This means we can simply implement the embedding module by indexing the embeddings matrix using the input batch: `output = embedding[input_batch]`

Like mentioned earlier, we'll initialize the embedding weights using `N(mean = 0, variance = 1)` truncated at `[-3, 3]`

Additional implementation instructions:

> Make sure to:
>
> - subclass `nn.Module`
> - call the superclass constructor
> - initialize your embedding matrix as a `nn.Parameter`
> - store the embedding matrix with the `d_model` being the final dimension
> - of course, don’t use `nn.Embedding` or `nn.functional.embedding`
> Again, use the settings from above for initialization, and use `torch.nn.init.trunc_normal_` to initialize the weights.

I've implemented the `Embedding` module class in [`nn_modules.py`](./nn_modules.py).

I've implemented the `run_embedding` function in [`tests/adapters.py`](../tests/adapters.py) to support testing.

To run official unit tests, run:

```sh
uv run pytest -k test_embedding
```

Got the following error

```
====================================================================== short test summary info ======================================================================
FAILED tests/test_model.py::test_embedding - AttributeError: cannot assign parameters before Module.__init__() call
============================================================ 1 failed, 47 deselected, 1 warning in 0.19s ============================================================
```

I think the issue is that I did not call `super().__init__()` in the `Embedding` constructor.

Yep, that was the issue, now the test passes:

```sh
uv run pytest -k test_embedding
======================================================================== test session starts ========================================================================
platform darwin -- Python 3.11.12, pytest-8.4.1, pluggy-1.6.0
rootdir: /Users/habbes/code/learn/stanford-cs336-llm-from-scratch/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                     

tests/test_model.py::test_embedding PASSED

================================================================= 1 passed, 47 deselected in 0.05s ==================================================================
```

Let's verify that the weights are properly initialized based on the expected distribution.

```python
>>> e = Embedding(1000, 500)
>>> e.weights.mean()
tensor(5.3624e-06, grad_fn=<MeanBackward0>)
>>> e.weights.std()
tensor(0.9872, grad_fn=<StdBackward0>)
>>> e.weights.max()
tensor(2.9999, grad_fn=<MaxBackward1>)
>>> e.weights.min()
tensor(-2.9996, grad_fn=<MinBackward1>)
```

This seems reasonable. Let me add a test for this in [`playground.py`](./playground.py)
under the function `test_embedding_module_initialization`

```sh
uv run python -m cs336_basics.playground
```

```sh
...
SCENARIO: Verify Embedding module weights are initialized with expected distribution
Test passed!
```

## 3.5 Pre-Norm Transformer block

Normalization is a technique for rescaling the weights of network during training to have predictable, stable range of values.
Instead of letting the values grow arbitrarily large or small, they are rescaled to have a stable mean and variance.

This helps address issues like vanishing gradients, which slow down gradient descent and learning. Also, during
training, the distribution of a layer keeps changing after applying the activations of the previous layer. As a
result, the model constantly has to "re-learn" to handle new input scales (this is called **internal covariate shift**).

Generally, normalization makes gradient descent and therefore learning easier, faster and more stable.

In the original Transfomer paper, each transformer block has two-sublayers: multi-head self-attention and position-wise feedfoward network.
There's a residual connection around each sublayer followed by layer normalization.

![Original transformer architecture](original-transformer-architecture.png)

This architecture is called **post-norm** since the layer norm is applied to the output of each sublayer. But recent
work has shown that applying normalization before the sublayers improves training stability. This is called "**pre-norm**".
The pre-norm transfomer is now the standard used in modern language models (e.g. GPT-3, LLaMA, PaLM, etc.). That's
what we'll use in this project.

![Pre-norm in transformer block](prenorm-transformer-block)

An intuition for pre-norm is that there is a clean “residual stream” without any normalization going from the input embeddings to the final output of the Transformer,
which is purported to improve gradient flow.

Related papers:
- [Batch Normalization: Accelerating deep network training by reducing internal covariate shift, 2015, Ioffe and Szegedy](Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift)
- [Layer Normalization, 2016, Ba et al (Introduces layer normalization as a better alternative to batch norm)](https://arxiv.org/abs/1607.06450)
- [Attention is all you need, 2017, Vaswani et al (OG Transformer architecture)](https://arxiv.org/abs/1706.03762)
- [Transformers without tears: Improving the normalization of self attention, 2019, Nguyen and Salazar](https://arxiv.org/abs/1910.05895)
- [On layer normalization in Transformer architecture, 2020, Xiong et al](https://arxiv.org/abs/2002.04745)

### 3.5.1 Root Mean Square Layer normalization

We're going to use `RMSNorm` activation:
Given a vector `a` with size `d_model` of activations, `RMSNorm` will rescale each activation `a[i]` as follows:

```
RMSNorm(a[i]) := (a[i] / RMS(a)) * g[i]
```

Where:

```python
RMS(a) := sqrt((1/d_model) * sum(i in 0..d_model: a[i]**2 + eps) )
```

Where `g[i]` is a learnable "gain" parameter (there `d_model` such parameters), and `eps` is
a hyperparameter that's often fixed to 1e-5.

The project instructions don't specify how to initialize the `g` vector, but the RMSNorm paper
provides a background of LayerNorm in chapter 3 and states that `g` is set to 1 at the beginning.

Related papers:
- [Root Mean Square Layer Normalization, 2019, Zhand and Sennrich (Introduces RMSNorm as an alternative to LayerNorm)](https://arxiv.org/abs/1910.07467)
- [LLaMA: Open and Efficient Foundation Language Models, 2023, Touvron et al (LLaMA uses pre-norm based on RMSNorm)](https://arxiv.org/abs/2302.13971)

**Note**: You should upcast your input to `torch.float32` to prevent overflow when you square the input. Overall,
your forward method should look like:

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# Your code here performing RMSNorm
...
result = ...
# Return the result in the original dtype
return result.to(in_dtype)
```

I've implemented the `RMSNorm` module in [`nn_modules.py`](./nn_modules.py).

Note that I played around with dimensions using `einops` to ensure the aggregations
were applied to each sample in the batch indepedently. For example, the sum of squares
should be applied independently to each sample in the batch, i.e. whereas the
input is `(batch_size, d_model)` the output of the sum squares should be `(batch_size, 1)`
such that there's a separate sum of squares for each item in the batch. The sqrt and
additions/mulitiplications by scalars are all element-wise, so no issue there.

So the RMS part should be `(batch_size, 1)`, and when it divides the input x, which is `(batch_size, d_model)`
Then for each sample in the batch, the activations in that sample will be divided by the corresponding
rms of that batch item. What definitely want to avoid having the same rms divide all elements input tensor
regardless of batch item. The output of this operation is `(batch_size, d_model)`.

Finally, when `(batch_size, d_model)` is multiplied by the vector g `(d_model,)`, then standard
PyTorch broadcasting applies since the last dimensions are the same. So it will do
position-wise multiplication per batch sample.

I've implemented the `run_rmsnorm` test adapter in [`tests/adapters.py`](../tests/adapters.py).

To run the test:

```sh
uv run pytest -k test_rmsnorm
```

Tests pass:

```
uv run pytest -k test_rmsnorm
======================================================================== test session starts ========================================================================
platform darwin -- Python 3.11.12, pytest-8.4.1, pluggy-1.6.0
rootdir: /Users/habbes/code/learn/stanford-cs336-llm-from-scratch/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                     

tests/test_model.py::test_rmsnorm PASSED

========================================================================= warnings summary ==========================================================================
tests/adapters.py:295
  /Users/habbes/code/learn/stanford-cs336-llm-from-scratch/assignment1-basics/tests/adapters.py:295: DeprecationWarning: invalid escape sequence '\T'
    """Given the weights of a Transformer language model and input indices,

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================================================ 1 passed, 47 deselected, 1 warning in 0.56s ============================================================
```

### 3.5.2 Position-Wise Feedforward Network

In the original Transformer paper, the feed-forward network consists of two linear transformations with ReLU between them.
The dimensionality of the inner feed-forward layer is typically 4x the input dimensionality.

Modern LLMs tend to incorporate 2 design changes:
- they use a different activation function
- employ a gating mechanism

We'll use the **SwiGLU** activation function, which combines **SiLU** (also called Swish) activation
with a "gating mechanism" called **Gated Linear Unit (GLU)**. We will also omit the bias terms sometimes used in linear layers,
following most modern LLMs since PaLM [Chowdhery et al., 2022] and LLaMA [Touvron et al., 2023].

```
SiLU(x) = x * sigmoid(x) = x / (1 + e**-x)
```

Geometrically, SiLU looks like ReLU but has a smooth curve at x=0, which makes it continous and differentiable,
unlike ReLU which is not continuous, and therefore not differentiable, at x=0.

![alt text](SiLU-and-ReLU-activation-functions.png)

GLUs were originally defined as the element-wise product of a linear transformation passed through
a sigmoid function and another transformation:

```
GLU(x, W1, W2) = sigmoid(W1x) * W2x
```

GLUs are suggested to "reduce the vanishing gradient problem for deep architectures by
providing a linear path for the gradients while retaining non-linear capabilities".

Putting SiLU and GLU together, we get the SwiGLU, which we will use for our feed-forward network

```
FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x)) * W3x)
```


Shazeer [2020] first proposed combining the SiLU/Swish activation with GLUs and conducted experiments
showing that SwiGLU outperforms baselines like ReLU and SiLU (without gating) on language modeling
tasks. Though we’ve mentioned some heuristic
arguments for these components (and the papers provide more supporting evidence), it’s good to keep an
empirical perspective: a now famous quote from Shazeer’s paper is

> We offer no explanation as to why these architectures seem to work; we attribute their success,
> as all else, to divine benevolence.


I should implement the SwiGLU feed-forward network

**Note**: in this particular case, you should feel free to use `torch.sigmoid` in your implementation
for numerical stability.

You should set dff to approximately (8/3) × d_model in your implementation, 
while ensuring that the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your
hardware.

I've implemented SwiGLU as the module `FFSwiGLU` in [`nn_modules.py`](./nn_modules.py). I used
the existing `Linear` module I implemented earlier for the linear layers.

For testing, I implemented the `run_swiglu` function in [`tests/adapters.py`](../tests/adapters.py).

To run the test:

```sh
uv run pytest -k test_swiglu
```

Tests passed:

```sh
uv run pytest -k test_swiglu
======================================================================== test session starts ========================================================================
platform darwin -- Python 3.11.12, pytest-8.4.1, pluggy-1.6.0
rootdir: /Users/habbes/code/learn/stanford-cs336-llm-from-scratch/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                     

tests/test_model.py::test_swiglu PASSED

================================================================= 1 passed, 47 deselected in 0.47s ==================================================================
```


