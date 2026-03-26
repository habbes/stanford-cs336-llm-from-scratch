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

