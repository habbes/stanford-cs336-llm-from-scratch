# 3. Transformed Language Model Architecture

Related papers:

- [Improving Language Understanding by Generative Pretraing, Radford et al 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

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
- Embedding: `NormalDistricution(mean = 0, variance = 1) truncated at [-3, 3]
- RMSNorm: 1

You should use `torch.nn.init.trunc_normal_` to initialize the truncated normal weights

### 3.4.2 Linear Module

In this part, I implement a linear module based on `nn.Module` to compute y = Wx, to mirror the standard [`nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.linear.Linear.html).

*Note*: Note that we do not include a bias term (i.e. b is conceptually 0 in y = Wx + b), following most modern LLMs.

