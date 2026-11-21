# 4. Training a Transformer LM

We now have the steps to preprocess the data (via tokenizer) and the model (Transformer). What 
remains is to build all of the code to support training. This consists of the following:

- **Loss**: we need to define the loss function (cross-entropy).
- **Optimizer**: we need to define the optimizer to minimize this loss (AdamW).
- **Training loop**: we need all the supporting infrastructure that loads data, saves checkpoints, and 
manages training.

## 4.1. Cross-entropy loss

I found the following videos by 3Blue1Brown timely in providing some intuitition behind entropy, cross-entropy
and why cross-entropy is used as loss function for language models:

- [Reinventing Entropy](https://www.youtube.com/watch?v=l6DKRf-fAAM)
- [But what is cross-entropy](https://www.youtube.com/watch?v=GlYgs6v2YfU)

The homework instruction defines the standard cross-entropy loss (negative log-likelihood) as follows:

![alt text](cross-entropy-loss-formula.png)

But instead of me just copying and implementing it blindly, I want to see whether I can use intuition and what I learned
from the 3Blue1Brown videos above to derive the loss formula from first principles.

Note that in some cases I may give hand-wavy, somewhat watered-down, explanations based on intuition.
Remember to consult the 3Blue1Brown videos and/or other sources for more rigorous and satisfying
explanations.

**Recap of Information Content, Entropy and Cross-Entropy**

Let's say we can construct messages from a vocabulary of symbols, where the probability of a symbol
appearing in the message is defined by the probability distribution P, i.e. `P[i]` is the probability
of symbol `i`.

Intuitively, the higher the probability of a symbol, the less we are surprised to see it in a message, similarly
the lower the probability, the more we are surprised to see it. The occurrence of low-probability event
provides more information than a high probabilty event, which is expected and easier to predict.

Intuitively, if we were to encode a message in binary and we want to minimize bit length (i.e. optimize compression),
it makes more sense to allocate more bits to less likely symbols, and fewer bits to more likely symbols. Since
the higher-probability symbols will be more frequent in the message, we should minimize their respective binary encoding
to minimize the average message size. This will lead to more efficient encoding (in terms of bytes allocated) than giving
each symbol the same number of bits. Conceptually, this is similar to how in BPE tokenization algorithm, we merge
the most frequent pairs into single tokens.

Now let's get into the maths a bit. It can be demonstrated that to optimize the compression of messages produced with
symbols based on some distribution P, the number of bits to allocate each symbol `i` should be `-log2(P[i])`. This
is considered the **Information content** of the symbol. An encoding based on this will lead to optimal compression.

Note that `-log2(P[i])` is equivalent to `log1/2(P[i])`, the number of times you have to chop something by half to get the probability.
It is a positive number that increases as `P[i]` decreases and is 0 when `P[i] == 1` (i.e. a symbol that is certain, constant doesn't
require any bits, such a message does not need to be transmitted since it's value is always known and never changes, transmitting it
is inherently redundant as it provides no new information).

And from this we can derive the average number of bits per symbol in a message M that uses symbols with distribution P,
i.e. the average information content of a message. This is just a weighted average of the number of bits per symbol,
where the weights are determined by the probabilities of each symbol. This is also called the **Entropy** and is
defined as:

```python
H(P) = -sum(P[i] * log2(P[i]))
```

Based on this, we expect that a message M using symbols based on distribution P, and using an optimal binary encoding
for that distribution will use on average `-sum(P[i] * log2(P[i]))` bits per symbol when measure over a large enough
sequence of symbols.

Note that the entropy assumes optimal encoding, i.e. an encoding that is optimized for distribution P, where
each symbol `i` is allocated bitlength based on `-log2(P[i])`. But what if we the symbols are based on a different distribution
`Q` but encoded using an encoding optimized for distribution `P`? Well, in this case we can use the **Cross-Entropy** to measure
the average bits per symbol of a message using an encoding optimized for distribution P, but where the symbols are actually
based on distribution Q:

```python
H(Q, P) = -sum(Q[i] * log2(P[i]))
```

The formula is similar to entropy, expect we use probabilities `Q[i]` for the weights since that represent the true distribution of the symbols,
but we use `log2(P[i])` for the bits per symbol since the encoding is based on P.

Now remember that an encoding based on `P` is optimal compression (minimizes average bit length) if the distribution of the symbols is also based on `P`. For other distributions, it will be sub-optimal. This means that `H(Q, P) == H(P)` if `Q == P` and `H(Q, P) > H(P)` when `Q != P`. The entropy `H(P)`
is the lower bound for the cross-entropy `H(Q, P)`, and this lowerbound is achieved when the two distributions are similar.

This means that the cross-entropy gives us a way to measure how similar the two distributions are. If the distribution Q is idential to P, then the
cross-entropy is minimized and is equivalent to the entropy of a message based on P.

**Application to language modelling**

Remember that a language model is not just predicting the next token, but generating a probability distribution for the next token (i.e. generating a probability for each possible token that can come next).

The task of the language model is to generate `P(next token | prefix)` for each possible token given some prefix sequence of tokens.
During training, we measure how well the generated probability distribution matches the underlying probability distribution of the training
dataset. And with a sufficiently large and well curated dataset, we want the distribution of the dataset is a good enough approximation
of the distribution of the language.

During training we want to use a loss function that is minimized when the model's output distribution matches the distribution of the training data,
and penalizes divergent distributions.

Now let's break it down to the level of training examples. During training we collect snippets from the training corpus where each
snippet is broken down into a prefix and target word/token. Let's take the following example of possible snippets that happen
to have the same prefix:

- I'm happy today
- I'm happy because
- I'm happy today
- I'm happy but
- I'm happy because
- I'm happy today
- I'm happy sometimes

If these are all occurrences of te prefix "I'm happy" in the training text, then the statistical distribution Q of the next
token based on the training data would be:

`P(next token| "I'm happy")` = [ Q_today: 0.43, Q_because: 0.29, Q_but: 0.14, Q_sometimes: 0.14]`.

Let's say the model outputs probability distribution P over the next token. For each example,
we use a loss function `f(x)` based on the output probability of the target token. The loss function
should penalize low probabilities of the expected target token (high loss) and reward high
probability of the target token.

So for the example "I'm happy -> today", we'll apply the loss function to the output probabilit P_today,
i.e. `f(P_today)`.

To get the average loss over all the training examples, we'll use a weighted average where the weight
of each loss will be based on the statistical probability of the target token in the training data:

```python
AverageLoss = sum(Q[i] * f(P[i]))
```

So what would be an appropriate loss function? Well for each example we want the loss to be minimized
when the probability of the target token is high (1), and the loss to be high when the output probability
of the target token is low. In addition, we want to average loss of all examples to be minimized
when the the output distribution of the target tokens matches the statistical distribution of the target tokens.
Specifically, the average loss should be minimized when the values of `P[i]` match the values of `Q[i]`.
We have already seen that the **cross-entropy** of P relative to Q matches this property, so cross-entropy
would be a good candidate for the average loss.

We can turn the formula above into cross-entropy by simply making the per-example loss function the negative log: `f(P[i]) == -log2(P[i])`.
As a result the average loss becomes the cross-entropy

```python
AverageLoss = sum(Q[i] * -log2(P[i])) = -sum(Q[i] * log2(P[i]))
```

Now there are many other possible per-example loss functions that have a similar shape as negative log (e.g. high when target probability is low, low
when target probability is high). The 2nd 3Blue1Brown video goes into more detail about why using a logarithm-based function is the ideal
option, I didn't get in the fine mathematical details.

For a single training example, we can consider that Q[target_token] == 1, i.e. in the snippet "I am happy today". Q(today | I am happy) is 1,
since that is the target token for that example, every other possible next token has prob 0.
So if only have a single example, the cross-entropy would be `-sum(Q[i] * log2(P[i]))`, since all `Q[i]` are 0 expect for `Q[target]`, most
terms will be eliminate leaving only `-Q[target] * log2(P[target])` = `- 1 * log2(P[target])` = `-log2(P[target])`. This is another
way to see how we can get from cross-entropy to the loss function.

Note that in deep learning we normally use `ln` (natural logarithm) instead of `log2` for the cross-entropy formula since `ln` is more
convenient to work with during gradient descent (since `d/dx ln(x) = 1/x`).

**Deriving training loss**

Now we know we'll use cross-entropy loss, we also know the per-example loss is `-log(P[target_token])`. Now we need to
apply this to the actual training process.

We'll be training one batch of data at a time. Each batch will have `B` sequences of `context_length` tokens.

I single sequence of size `m` will produce `m-1` training examples, e.g.

The sequence "I am happy today" will produce the following examples (assuming each word is a separate token):

Prefix | Target
-------|---------
I | am
I am | happy
I am happy | today

So for a sequence `x` of size `m + 1`, we'll produce `m` training examples.

In each example, the prefix is the range `x[1:i]` (inclusive) and the target is `x[i + 1]`, where
`i` goes from 1 to `m`.

The model will produce probability the distribution `P(V[k] | x[1:i])` where `x[1:i]` is the sequence of tokens from 1 to i (inclusive).
`V` is the vocabulary vector and `k` goes from 1 to `vocab_size`.

The loss function for this specific example is is based probability assigned to the target token `x[i + 1]`.

So the loss for the example is 

```python
LossForSingleExample =  -log(P(x[i + 1] | x[1:i]))
```

Now the average loss for all `m` training examples in the source sequence `x` will be:

```python
AverageLossForExamplesInSequence(x) = (1/m) * sum(-log(P(x[i + 1] | x[1:i])) for i in range(m))
```

Since we'll be training for each sequence in the training set, we need to get the average loss
over the entire data set (or in this case the batch, since we're doing mini batch gradient descent).
So we'll just aggregate the average of the loss above by the size of the dataset or batch:

```python
AverageLoss = (1/len(dataset)) * sum(AverageLossForExamplesInSequence(x) for x in dataset)
```

```python
AverageLoss = (1/len(dataset)) * sum((1/m) * sum(-log(P(x[i + 1] | x[1:i])) for i in range(m)) for x in dataset)
```

We can extract that `1/m` constant factor out of the outer summation to get:

```python
AverageLoss = (1/len(dataset)) * (1/m) * sum(sum(-log(P(x[i + 1] | x[1:i])) for i in range(m)) for x in dataset)
```

And this is equivalent to the original loss function formula in the instructions

![alt text](cross-entropy-loss-formula.png)

Now remember that the transformer architecture we implement outputs unnormalized logits in the final
layer. The output layer has dimension `context_length * vocab_size` (i.e. output vector of vocab size for each of the context-length sequences),
assigning an unnormalized weight to each possible
token in the vocabulary. We turn these unnormalized weights in a probablity distribution using `softmax`:

```python
P[x[i + 1] | x[1:i]] = softmax(O[i])[x[i + 1]] = (e**O[i][x[i + 1]]) / sum(e**O[i][j] for j in range(vocab_size))
```

Where `O[i]` is the output vocab-sized vector correspoding to the training sequence `x[1:i]`, whose target output is the token `x[i + 1]

### 4.1.a Implementing cross-entropy loss

Implementing the cross-entropy loss requires some care with numerical issues, just like in the case of 
`softmax`.

> **Deliverable**: Write a function to compute the cross-entropy loss, which takes in predicted logits `o[i]`
> and targets `x[i + 1]` and computes the cross-entropy `l[i] = -log * softmax(o[i])[x[i + 1]]`.
> Your function should handle the following:
>
> - Subtract the largest element for numerical stability
> - Cancel out `log` and `exp` whenever possible
> - Handle any additional batch dimensions and return the average loss across the batch. As with section 3.2 we assume batch-like dimensions always come first, before the vocabulary size dimension.
>
> Implement`adapters.run_cross_entropy`, then run `uv run pytest -k test_cross_entropy` to 
> test your implementation.

Based on the hints above, we can transform the `-log * softmax(o[i])[x[i + 1]]` for better numerical stability, avoid over and underflows.

Here's the expanded version of the formula:

![alt text](simplify-cross-entropy-loss-for-numerical-stability.png)

```python
exps = exp(o[i])
l[i] = -log * ((exps[x[i + 1]]) / sum(exps[j] for j in range(vocab_size)))
```

First, like in softmax, we can subtract the largest values (review `softmax` implementation to see why this works):

```python
exps = exp(o[i] - max(o[i]))
l[i] = -log((exps[x[i + 1]]) / sum(exps[j] for j in range(vocab_size)))
```

Now, remember the logarithmic identity `log(a / b) = log(a) - log(b)`. We can use
this to transform the formula above to:

```python
l[i] = - (log(exps[x[i + 1]]) - log(sum(exps)))
# which turns to
l[i] = -log(exps[x[i + 1]]) + log(sum(exps)))
```

Since `log(e**x) == x`, we can cancel out the `log` and `exp` in the first term. We can't
really simplify the second term though since it's a log of sums (and not a log of products).

```python
l[i] = -x[i + 1] + log(sum(exps))
```

### 4.1.b Perplexity

Cross-entropy suffices for training, but when we evaluate the model, we also want to report perplexity. 
For a sequence of length `m` where we suffer cross-entropy losses `l_1`, `l_2`, ..., `l_m`:

![alt text](perplexity-formula.png)

Basically, perplexity is the exponentiated average cross-entropy loss across our tokens.
We can also think of it as:

```python
perplexity(x, y) = exp(cross_entropy(x, y))
```

But what does this mean, why do we exponeniate the average cross-entropy/negative-likelihood loss?
Well an increase and decrease in cross_entropy also increases or decreases perplexity respectively,
but at different scales. It's harder to get an intuition of differnet cross-entropy values because
comparing logarithms is not very intuitive for humans. Perplexity "undoes" the logarithm
to make the loss easier to interpret, transforming the loss from a logarithm scale to a linear scale.


Intuitively, perplexity measures the performance of a model based on how confused it is.
Mathematically, it's also equivalent to the inverse of the probability assigned to the correct next token. It
tells how likely it was to pick the correct token based on a compressed number of choices.

Mathematically

Let `P` be the probability assigned to the correct next token by the model:

```python
H = -ln(P)
PPL = exp(H) = exp(-ln(P)) = 1/(exp(ln(P))) = 1/P
```


Let's take the following example:

The model assigns a 10% `0.1`probability to the correct
next token.

The cross-entropy loss `l = -log(0.1)` is  `2.3026`.

The perplexity is `exp(2.3026)` which is equivalent to `1/0.1` which is `10`.

A perplexity of 10 means the correct token had a 1/10 chance of being picked.
It tells you the model is guessing as if it were choosing between 10 equally likely options.

So if the vocab has 50,000 tokens, a perplexity of 10 means the model is performign really well.
A random guess would have a perplexity of 50,000.By achieving a perplexity of 10,
the model has used context to compress a massive dictionary of 50,000 possibilities down to a tiny
shortlist of just 10 realistic options.

A perplexity of 1, means the model is always certain of the next token.

A logarithmic scale compresses huge differences into small numbers. Small changes in cross-entropy
loss represent massive changes in how smart the model is. By exponentiating the loss, perplexity
undoes the that compression and converts the logarithmic scale back into a standard, linear scale.

Let's say Model A has a cross-entropy loss of 4.6 and Model B has a cross-entropy loss of 3.0.
The 1.6 difference may seem deceptly small. But the perplexity of A is 100 and B is 20. This
makes it clear to see that B is 5x more capable.

## 4.2. The SGD Optimizer

Next, we begin exploring optimizers. Optimizers update the value of learnable parameters based on the loss. A gradient-based optimizer
updates each parameter based on the gradient of the loss with respect to the parameter.

The simplest gradient based optimizer is Stochastic Gradient Descent (SGD). We start with randomly initialized parameters `theta[0]`.
Then for each step *t = 0, ..., T-1*, we perform the following update:

```python
theta[t + 1] = theta[t] - alpha[t] * gradient(loss(theta[t], B[t]))
```

![alt text](sgd_update_formula.png)

Where:
- `B[t]` is a random batch of data sampled from the dataset D
- `alpha` is the learning rate

The learning rate and batch size are hyperparameters.

### 4.2.1 Implement SGD in PyTorch.

To demonstrate the `Optimizer` API in PyTorch (base class used to implement custom optimizer), we implement
a modified variant of SGD where the learning rate decays over time based on the formula:

![alt text](sgd_with_decaying_learning_rate.png)

I've implement this optimizer in [`sgd_optim_example.py`](./sgd_optim_example.py).

#### Learning rate tuning

> As we will see, one of the hyperparameters that affects training the most is the learning rate.
> Let’s see that in practice in our toy example. Run the SGD example above with three other values 
> for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss
> for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over 
> the course of training)?
>
> **Deliverable**: A one-to-two sentence response with the behaviors you observed

To answer this questions, I added the scenarios to `sgd_optim_example.py`, then run as:

```sh
uv run python -m cs336_basics.sgd_optim_example
```

```sh
Running training loop with LR=1 and num iterations=10
22.424942016601562
21.536914825439453
20.932065963745117
20.45145034790039
20.04446792602539
19.687503814697266
19.367321014404297
19.075620651245117
18.806804656982422
18.556882858276367
Done.

Running training loop with LR=10.0 and num iterations=10
25.632505416870117
16.404802322387695
12.092920303344727
9.461418151855469
7.663748741149902
6.3541259765625
5.358862400054932
4.579300880432129
3.9545862674713135
3.4448840618133545
Done.

Running training loop with LR=100.0 and num iterations=10
29.442251205444336
29.442249298095703
5.0514912605285645
0.12089355289936066
2.01826404930104e-16
2.249480452710394e-18
7.574792075885271e-20
4.512355867594862e-21
3.8709906965819494e-22
4.301100949282367e-23
Done.

Running training loop with LR=1000.0 and num iterations=10
23.219282150268555
8382.1611328125
1447731.25
161044608.0
13044613120.0
823264477184.0
42263702929408.0
1818365404381184.0
6.702107057350246e+16
2.1521211131680522e+18
Done.
```

Answer:

When learning rate is 1e1 the loss decays much faster than with lr=1. When lr is 1e2,  the loss quickly decays
to near 0 then starts to oscillate erratically above the minimum. When lr is 1e3 the loss diverges to very large values.

## 4.3 AdamW Optimizer

AdamW is a stateful optimizer that keeps track of running estimates of the first and second moments for each parameter.
Before implementing it, I'm curious to understand the rationale behind the evolution from SGD to AdamW and
the optimizer's algorithm.

AdamW is one of a number of adapative optimizer algorithms, we can trace the evolution from **SGD** -> **SGD with momentum** -> **AdaGrad** -> **RMSProp** -> **Adam** -> **AdamW**.

Standard SGD uses the same learning rate to update all parameters. And at each time step
SGD uses only the current gradient to determine the magnitude and direction of the update. SGD with momentum combines the current
gradient with a running velocity based on previous gradients.

**SGD**:

```python
param[t + 1] = param[t] - learning_rate * gradient[t]
```

**SGD with momentum**:

```python
v[t] = beta * v[t - 1] + gradient[t]
param[t + 1] = param[t] - learning_rate * v[t]
```


The other algorithms are a bit more complex but are also based on the idea
of maintaining state across timesteps and adjusting how each parameter is updated instead of relying only on the current gradient
and a global learning rate. Adam aims to combine the benefits of AdaGrad and RMSProp, and AdamW is a variant of Adam that makes weight decay more
effective. So I'll start with with a review of the Adam optimization algorithm.

### 4.3.1 Adam optimizer

**Source: [Adam: A method for stochastic optimization, 2015](https://arxiv.org/pdf/1412.6980)**

The name Adam is derived from "adapative moment estimation". It was proposed as a method for "efficient" stochastic optimization
that only requires first-order gradients (i.e. first derivatives only, no need for 2nd or higher derivatives) with "little memory
requirement". Note that here the "little memory requirement" is relative to approaches that computer higher-order derivatives. Adam
stores extra state for each parameter, so it requires O(n) memory with respect to model parameter count, so this is definitely less efficient
than standard SGD. But consider efficient compares to methods with require quadratic memory for example.

The Adam optimization algorithm has the following goals/benefits:

- Support for online and stochastic environments: i.e. loss function is not computed against entire dataset at once, but one at a time, or in minibatches
- Support for noisy objective functions: noise might be due to the fact that objective function is computed on different batches of data, or inherent to the function itself
- Support for non-stationary data: the distribution of the data might change across batches of data (i.e. mean, variance, and other characteristics of the dataset may change from on batch to the other)
- Robust to noisy and sparse gradients: should adapt to sudden gradient spikes, should adapt to cases where gradient is 0 most of the time
- Robust to gradient rescaling: if step size is directly proportional to gradient (like in SDG), then a sudden scaling factor of the gradient can cause the optimizer to take massive,
    exploding steps in parameter space, making paramter updates unstable and ruining training. Sensitivy to gradient scale also means changing something seemingly unrelated, like scaling the loss function or batch size, forces you to retune learning rate from scratch.

In short, Adam aims to adapt to a chaotic optimization problem while still being relatively computationally efficient.

It computes adaptive learning rates for each parameter from estimates of the first and second moments of the 
gradients. The first moment refers to the mean (mean of g) and the second moment to the variance (mean of g**2) 
(in this case uncentered variance since the mean is not subtracted from the squared gradients like in "standard" variance).
Adam doesn't compute the actual mean and variance, which would require storing the full history of gradients and would be inefficient, but only estimates using the last computed value.

#### 4.3.1.a. Attempt at deriving Adam algorithm

The following is my attempt at deriving the Adam formula from my understanding.

Let's review the SGD update rule again: `param[t + 1] = param[t] - lr * g[t]`.

As mentioned, the update only takes the current gradient value (adjusted by the learning rate) into account. This is prone
to noisy updates if the gradient is temporarily unstable (changes direction quickly). It's also sensitive to large changes
in the magnitude of the gradient.

Conceptually, Adam updates the parameter based on the average gradient, rather than just the current gradient. It also
normalizes the average gradient by the average magnitude of the gradient to reduce sensitivity to gradient's scale.

![alt text](adam-conceptual-formula.png)

Conceptually, Adam is an approximation of `param[t + 1] = param[t] - lr * E(g[t])/sqrt(E(g[t]**2))`

`E(g[t])` here means the expected gradient at timestep `t`. Or the average/mean of the gradient. Using
the average rather than the latest value of the gradient at timestep `t`, makes the updates more robust to noise
and sudden spikes.

`E[g[t]**2]` is the expected variance. As meantioned before, we use the uncentered variance here. This estimates
the average maginute or "size" of the gradient values (note that gradient values can be positive or negative, but magnitude is always positive).
We use this to normalize/divide the scale of the average gradient. Since the gradient is linear, and variance is squared, we take the
square root of the variance to bring it to the same scale as the average gradient so the division can be stable. This normalization
by the variance makes the updates more robust to gradient rescaling.

In machine learning optimization papers, we often use `m` to refer to the first moment, i.e. mean, and `v` to refer to the second
moment, i.e. variance.

```python
m = E(g)
v = E(g**2)
```

So we can rewrite the formula above as:

```python
param[t + 1] = param[t] - lr * m[t]/sqrt(v[t])
```

To prevent a 0 denominator when `v[t]` is 0, we can add a small neglibile value, epislon, for numerical stability:

```python
param[t + 1] = param[t] - lr * (m[t]/(sqrt(v[t]) + eps))
```

![alt text](adam-conceptual-formula-with-eps.png)

If we were to stop at this point, then we would have some level of resilience to noise and gradient rescaling, but what about sparse gradients, non-stationary data?
If the update is based on the global mean of all the data seen so far, then it means the current gradient has the same contribution to the mean as each gradient seen in the past.
This means the more data points and batches we visit, the smaller the impact of the latest gradient to the mean and therefore to the update. The oldest gradient
seen has just as much impact on the current update as the current mean. So the latest gradient may become relatively useless after seeing enough samples.
This would be fine if the gradient is mostly uniform or has a stable distribution across batches, but if the distribution changes or is too noisy/erratic, relying on the global
means the optimizer will not be able to adapt or will take too long to adapt to changes in distribution. If the data has sparse gradients, e.g. parameters in the model
that are activated by a unique data points that are relatively rare in the dataset, then the mean will be close to zero and a single non-zero gradient over multiple datapoints may not have enough
impact to tilt the mean enough for a meaninful update.

Adam addresses these concerns by using weighted averages instead of standard mean where every gradient has the same weight. Specifically, Adam uses exponential moving averages where
past gradient history decays exponentially. This allows the optimizer to recover from sparse gradients or adapt to changing distribution reasonably quickly, while still being reasonably
resilient to localized noise.

Instead of allocating past gradients individual weights, which would require us to store past gradients in memory, we use the following formulation which is more efficient estimate
for the exponetial moving average:

```python
m[t] = beta * m[t-1] + (1 - beta) * g[t]
```

Where `beta` is a value between [0-1]. This basically determines how much weight the current gradient has on the current parameter versus historical gradients. The closer beta is to 1,
the less affected it is by the current gradient and the smoother/less noisy m[t] is. The closer beta is to 0, the faster it can adapt to changes in distribution, sparsity, etc.

Note that each step, the weight a specific past gradient decays exponentially. To demonstrate this, let's track the weight of `g[2]` across of a few timesteps, assuming beta = 0.9.

At t = 2, `m2 = beta * m1 + (1 - beta) * g2`, the weight of `g2` is `1 - beta` = 0.1

At t = 3, `m3 = beta * m2 + (1 - beta) * g3` = `beta * (beta * m1 + (1 - beta) * g2) + (1 - beta) * g3`.
If we extract out `g2`, its weight is `beta * (1 - beta)` == `(beta - beta**2)` = `0.9 - 0.81` = `0.09`.

At t = 4, `m4 = beta * m3 + (1 - beta) * g4` = `beta * (beta * (beta * m1 + (1 - beta) * g2) + (1 - beta) * g3) + (1-beta) * g4`
Let's extract g2 from `beta * (beta * (beta * m1 + (1 - beta) * g2)`  = `beta * ((beta - beta**2))` = `beta**2 - beta**3` = (0.81 - 0.729) = 0.0081

![alt text](adam-g2-weight-breakdown.png)

The pattern is at t, g2's weight is `beta**(t-2) - beta**(t-1)`, so at t = 5, the weight is 0.0729.

![alt text](adam-g2-weight-pattern.png)

 t | g2 weight
---|-----------
2  | 0.1
3  | 0.09
4  | 0.081
5  | 0.0729
10 | 0.03874
20 | 0.01351
50 | 0.000573

The weight decays slowly at first then eventually tapers off where it gets so close to 0 it's practically irrelevant to the current update.
With a starting weight of 0.1, the gradient still retains most of its weight in the next couple of time steps. The local neighbors
retain considerable influence to the current update compared to much older timesteps.

![alt text](adam-beta1-exponential-weight-decay-graph.png)

The exponential weighted average allows the optimizer to adapt to non-stationary data, noisy functions, sparse gradients, etc.

We'll also apply an exponential weight average to the variance, consequently we'll have beta_1 for the mean `m` and beta_2 for the variance `v`:

```python
m[t] = beta_1 * m[t-1] + (1 - beta_1) * g[t]
v[t] = beta_2 * v[t-1] + (1 - beta_2) * g[t]**2
```

Here's what the weights for `g2**2` look like when `beta_2 == 0.999`

 t    | g2 weight
------|-----------
2     | 0.001
10    | 0.00099
500   | 0.00061
1000  | 0.00037
2000  | 0.00014
4000  | 0.000018
6000  | 0.0000

![alt text](adam-beta2-exponential-weight-decay-graph.png)

Based on the update formula above, we can imagine the Adam algorithm to look as follow:

```python
def update_param(param, lr, beta_1, beta_2, eps, has_converged):
    m = 0
    v = 0
    for not has_converged()
        g = get_gradient(param)
        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * m + (1 - beta_2) * (g**2)
        param.data -= lr * (m/(sqrt(v) + eps))

```

The Adam proposes as good defaults `lr = 0.001`, `beta_1 = 0.9` and `beta_2 = 0.999` and `eps = 10**-8`.
Note that `beta_2` is larger than `beta_1` by 2 orders of magnitude, this means individual variance has much lower
weight. Squaring gradients makes them much more noisy, small fluctuations in the gradient lead to much larger changes
in the variance (squared gradient). So we need a much small weight to keep a noisy spike from having too much influence
on future updates.

In statistics, there's a formula used to calculate the "center of mass" or effective lookback window of an exponential moving average:

```python
window_size = 1 / (1 - beta)
```

Conceptually, this tells how many steps really matter in influencing the average before the influence starts to taper off into irrelevance.

- For first moment, beta_2 = 0.9, 1/(1 - 0.9) = 10 steps
- For second moment, beta_2 = 0.999, 1 / (1 - 0.999) = 1000 steps

Adam tracks the gradient's *direction* over a short window that's quick to adapt to local twists and turns,
but tracks the *spread/variance* over a massive, ultra-smooth window. If `v` had a short memory (like 10 steps),
a single freak batch would cause `v[t]` to shoot straight up, and because `v[t]` lives in the denominator,
a sudden spike would crush the learning rate for the next several steps, freezing progress.
Therefore, the heavy smoothing ensures that `v[t]` acts as a stable, reliable denominator. It absorbs high-frequency
noise and only shifts slowly when the true underlying magnitude of the gradients changes over a long horizon.

We can also corroborate this by looking at the aggregate weight at 10 steps for b1 = 0.9 and 1000 steps at b2 = 0.999.

The aggregate weight of the last n timesteps can be computed by

![alt text](adam-aggregate-weight-of-last-n-steps-breakdown.png)

```python
aggregate_weight_of_last_n_steps = lambda n: sum(beta**k * (1-beta) for k in range(0, n))
```

For beta1 = 0.9, n = 10 steps, the aggregate weight is 0.65

For beta2 = 0.999, n = 1000 steps, the aggregate weight is 0.63

The aggregate weights at the respective effective window sizes are almost the same value (0.65 and 0.63)

Now let's dive back to the algorithm we have so far:

```python
def update_param(param, lr, beta_1, beta_2, eps, has_converged):
    m = 0
    v = 0
    for not has_converged()
        g = get_gradient(param)
        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * m + (1 - beta_2) * (g**2)
        param.data -= lr * (m/(sqrt(v) + eps))

```

This looks good in general based on everything we've discussed so far, however it has one critical issue.

Based on the formula `beta * m[t-1] + (1 - beta) * g1`, what we're trying to achieve is for 10% of the weight of the average to come from the current gradient, and 90% from past gradients.

However, we initialize `m` and `v` to 0, which means at `t_1`, we have `m[1] = beta_1 * 0 + (1 - beta_1) * g = (1 - beta_1) * g`, i.e. the first update has no history. Since the first
term is 0, `g_1` is the only gradient influence for the first step:

```python
m1 = 0.9 * 0 + 0.1 * g1 = 0.1g1
```

Now let's see the weight distribution between current and past gradients for the next couple of steps:

```python
m2 = 0.9 (0.1 * g1) + 0.1*g2 = 0.09g1 + 0.1g2

m3 = 0.9 (0.09g1 + 0.1g2) + 0.1g3 = 0.081g1 + 0.09g2 + 0.1g3 = 0.0171(past weights) + 0.1g3

m4 = 0.9 (0.171 + 0.1) + 0.1g4 = 0.2439(past weights) + 0.1g4

m5 = 0.9(0.2439 + 0.1) + 0.1g5 = 0.30951(past weights) + 0.1g5
```

In step 1, the weight of past gradients 0 and average is determined exclusively by the current gradient,
in subsequent steps the weight of the past gradient slowly starts to increase, but is still much lower than the expected 0.9.
This mean that the weight is more skewed towards the current gradient than we desire.
For example, at step 5 with 0.30951 as the aggregate weight of past gradients,
then the effective weights are actually 0.30951/(0.30951 + 0.1) and 0.1/(0.30951 + 0.1) which are 76% and 24% respectively.

So my question then is, after how many steps do we arrive at a steady state where the weight distribution is close
to the desired 90%-10% split between past gradients and current gradient.

To answer this, let's start with how to compute the aggregate weight of all previous steps, based on the formula we already derived above:

![alt text](adam-aggregate-weight-of-previous-steps.png)

```python
agg_weight_of_prev_steps = (1 - beta) * sum(beta ** k, for k = 0..t)
```

Let's try to factor out the summation to get a simplified formula:

![alt text](adam-aggregate-weight-simplification.png)

We get:

```python
agg_weight_of_last_gradients = 1 - beta**t
```

Note that this is the weight of the current gradient `g[t]`. To exclude
the current gradient's weight, we can adjust the formula:

![alt text](adam-aggregate-weight-exclusive.png)

```
agg_weight_of_last_gradients_exclusive = beta - beta**t
= beta * (1 - beta**(t-1)
``` 

Now I want to know, how many steps do we need until we get to aggregate weight of 0.9,
i.e. assume the aggregate weight is `w` (exclusive of current gradient) given by the formula:

```python
w = beta - beta**t
```

Now we want to solve for `t` to get a target value of `w`:

![alt text](adam-number-of-steps-to-target-exclusive-weight.png)

```python
t = ln(beta - w)/ln(beta)
```

Notice, if we set target weight `w == beta`, total result witll be `Infinity`. This because
we never really reach the target aggregate weight. If `beta` is `0.9`, the target weight
will grow infinitely closer to `0.9` but will never reach it. We're adding
a sequence of numbers that are exponentially smaller. `0.9` is the limit.

Since we can reach beta, let's pick a close a enough target weight, let's say
within a percent of the target weight (`0.9 * (1 - 0.1) == 0.891`).

So for:

- For beta = 0.9 and w = 0.891, t = 45
- For beta = 0.999 and w = 0.98901, t = 4603

So it would take rougly 45 steps to get to within 1% of the target weight distribution for the first moment, and 4603 steps for the second moment.
That's a lot of training iterations before we get to desired state! Bear in mind that at mt[1], the previous step is 0, the value
is 0.001 * (g**2), which could be a really small value. This would make the denominator of the update formula close to 0.
This might cause the learning rate to blow up or behave erratically.

#### 4.3.1.b My attempt at deriving Adam's initialization bias correction

**Note**: *This attempt has a fundamental flaw which I return to revise after explaining the paper's approach*.

This sets up the stage for the **initialization bias correction** solution. But before I head back to the paper,
I want to see if I can figure out a bias correction solution on my own, the compare with the paper.

Here's my attempt:

Given the base formula `m[t] = beta * m[t-1] + (1 - beta) * g[t]`, what I want to achieve is to rescale the weights
for `m[t-1]` and `g[t]` so that they match the desired weights of `beta` and `1-beta` from the early steps. So in case of `beta = 0.9`,
I want the weight of previous gradients to be `0.9` and the weight of `g[t]` to be 0.1 from the beginning, without
have to weight for 45+ iterations.

Since the weight of previous gradients is much lower in early iterations, my initial intuition is that we either need to increase its weight
in early iterations, or decrease the weight of `g[t]`. I will go with former, i.e. attempt to increase the weight of `m[t-1]`.

Let's start with `t = 2`, we already found that

```
m[2] = 0.9 * m[1] * 0.1g[2] = 0.9 * 0.1g[1] * 0.1g[2] = 0.09 * g[1] + 0.1g2
```

i.e. the weight of previous gradients is `0.09`. But we want to scale this to `0.9`. Well we can simply
do that by multiplying by 10 `0.09 * 10 = 0.9`. But we want to generalize the scaling factor. In fact,
we want a scaling factor that decreases with each time step, and at each step it should multiply the weight of the previous
gradients so that the output is approx. `beta`.

At m[1], we don't have a previous gradient, so let's skip this.
At m[2] the weight of previous gradients is 0.09, we want to scale it to 0.9, so the scaling factor is 10.

We expect the scaling factor to reduce and get to 1 around step 44-45 based on the calculation we did to find the step where we reach the target weight.

So we're looking for a function such that:

f(2) = 10
f(45) = approx 1

How did we get 10 at step 2? well:

![alt text](adam-t2-scaling-factor-breakdown.png)

So:



f(2) = 1/(1 - beta)
f(ln(1-beta)/(1-beta)) = approx. 1


So this gives us a hint that formula should have some form of (1 - beta) in the denominator, but should be close to 1 at the natural steady state step.
For the formula to be 1, 1-beta should be 1, so the `beta` term should actually be a term that decreases with each step. Intuitively I feel
it should decay exponentially, given that we're dealing with an exponential moving average. So what if we raise `beta` to a value proportional to the step.

Since at t2, we have `beta` in the denominator, then perhaps we should raise to `beta ** (t-1)`. So at t2, we'll have beta, at t3 we'll have beta ** 3. At t=1, this is undefined because there are no previous gradients.

So the formula would be:

```
f(t) = 1/(1 - beta ** (t-1))
```

So this gives us that factor to multiply the previous gradients by so that we can get to 0.9:
Let's test this with values we computed for the first few steps:

step | w = orig. weight of prev steps | f(t) = 1 / 1 - beta ** (t-1) | new weight = f(t) * w
-----|--------------------------------|------------------------------|------------------------
2    | 0.09                           | 10                           | 0.9
3    | 0.171                          | 5.263                        | 0.9
4    | 0.2439                         | 3.69                         | 0.9
5    | 0.30951                        | 2.9                          | 0.9

It seems to check out so far,

Perhaps a better way to think about is, what do we need to multiply to the initial weight of previous steps to get beta.

![alt text](adam-initial-bias-scaling-factor-derivation.png)

The above equation shows that if the total aggregate weight of previous steps at timestep step is `beta * (1 - beta ** (t-1))`
then, the scaling factor `f` is what multiplies this to get `beta`, and that turns out to be `beta / (beta * (1 - beta ** (t-1)))`,
which is the same formula we got above:

```
f(t) = 1 / (1 - beta ** (t-1))
```

So the moment formula (starting for m[2]) would then be:

```
m[t] = (1 / (1 - beta ** (t-1))) * beta * m[t-1] + (1-beta)g[t]
``` 

Let's see if this holds up for a few iterations:

![alt text](adam-bias-correction-attempt-test.png)

We see that for `m[2]` it yields the expected distribution:

```
m[2] = 0.9g1 + 0.1g2
```

However, in `m[3]` the aggregate weight of gradients g1 and g2 blows up:

```python
m3 = 4.263g2 + 0.474g1 + 0.1g3
```

What went wrong?

The key mistake here is that the formula I came up with assumes that we're multiplying the bias
rescaling factor to what the original weight would have been without bias correction.
This assumption holds for `m2` because we apply the bias factor to the original `m2` formula.
However, it does not work for `m3` because the `m2` term in the `m3` formula already has
the rescaling factor applied to `g1`. This makes the weight of `g1` an order of magnitude
higher than it should be.

Consequently, the formula does not work for recursive scenario, where we are expected
to plug the result of the previous iteration (m[t-1]) in the current iteration's computation.

I need away to get the scaling to be spread gradually and recursively to all previous steps.

At this pointed i decided to just consult the approach taken by the paper.

#### 4.3.1.c Adam's Initialzation Bias Correction from the paper

The Adam paper takes a different approach at correcting the initialization bias. My attempt
had fundamental flaws, but still derived some useful terms that make it easier to understand
the paper's derivation.

**Note**: *The paper uses the second moment `v` to derive the correction, but the same concept works for the first moment `m`.*

First of all, the paper mentions that main flaw of the original moment estimate formula:
`v[t] = beta2 * v[t-1] * (1 - beta2) * g[t]**2`

Since `v[0]` is initialized to a zero vector, the moment estimates in the first are biased towards 0
especially in the first timesteps.
E.g. if `beta2` is `0.999`, then `v[1] = 0.0001 * g[t]**2` and, `v[2] = 0.999(0.0001) * g[1] + 0.0001 * g[2]`.

As you can see, since `v[1]` starts with small weight for `g1`, this small weight carries over and gets
attenuated in `v[2]` and the small amount of `g[2]` added won't make up for it. So it will take while
for the gradients to build up.

The larger `beta2` is (i.e. the smaller the decay rate), the more this issue manifests. The issue is exarcebated
when you have sparse gradients, where most gradients are zero. In this case you have to average over a large
number of gradients, i.e. a larger effective window which implies a larger `beta2`, which as we've mentioned
suffers more from zero-bias in the initial timesteps.

**Note**: *The paper does not concern itself, or at least does not mention, maintaining a specific split between current and past gradients' weights as an objective to be concerned with. My fixation with that "issue" might have been the biggest flaw with my earlier attempt*.

To derive a correction for the bias, the paper focuses on the relationship or discrepancy between the estimate of the second moment at timestep t
from the exponential moving average v[t] to the true second moment.

Since v[t] is an estimate of the second moment at time t, we want to know how the expected
value of v[t], `E[v[t]]` relates to the true second moment `E[g**2]`. Specifically,
we want to determine the discrepancy between the two and account for it.

In the earlier attempt, we already found that the exponential average equation:

```
v[t] = beta * v[t-1] + (1-beta2) * g**2
```

can be expanded to the following summation:

```python
v[t] = (1 - beta2) * sum(beta2**(t-i) * g[i]**2 for i in [0..t-1])
```

We start by taking the expectation of each side of the equation:

![alt text](adam-variance-estimate-expectation-breakdown.png)

No we have an expression based on the sum of expected values of the second moment at
different timestep. If the distribution of the gradients is globally stationary,
then we would expect the expected value of the second moment to be the same at each
timestep t. If it's not truly globally stationary, we should expect a discrepancy
between the expected value at time t `E[g[t]**2]` and the expected value at some
other time step i `E[g[i]**2]`. We can express this discrepancy as some
`error[i] = E[g[i]**2] - E[g[t]**2]`. And therefore `E[g[i]**2] = E[g[t]**2] + error[i]`.

Since this error is a constant for each `i`, we can just accumulate all the errors into a single constant
that we can extract out of the summation. And since `E[g[t]**2]` is also a constant with respect
to `i`, we can also extract it from the summation.

![alt text](adam-expected-variance-constant-extraction.png)

And now we have the expression `(1-beta2) * sum(beta2**(t-i) for i in range [1..t-1])`
which we already found can be simplified to `1 - beta2**t`.

Now, as we mentioned above, if the distribution is stationary, then `E[g[t]**2]` would be the same
as `E[g[i]**2]`, and therefore the discrepancy term would be 0. But even if the distribution
is not globally stationary, the algorithm expects us to select a value of `beta2` such that
the local window around timestep `t` is stationary, and older gradient values which are outside the
local distribution around `t` have insignificant weight. So we can assume that the discrepancy
term is near zero.

As a result, the estimated value of the exponential moving average `E[v[t]]` at time `t` is different
from the true second moment `E[g[t]**2]` roughly by a factor of `(1 - beta2**t)` which arises
from the fact that the exponential moving average starts with a zero vector.

![alt text](adam-estimated-variance-expected-variance-equation-derived.png)

And we have derived the bias correction: we simply divide the moment estimate `v[t]/(1-beta2**t)`

So we'll create new variables to correct the estimates:

```python
m_hat[t] = m[t] / (1 - beta**t)
```

```python
v_hat[t] = v[t] / (1 - beta2**t)
```

Let me test this by computing the weights of a few iterations. I'm going to use the first moment `m` instead
of `v` so that I can compare the values with results I got when I attempted to derive the bias correction:

![alt text](adam-bias-correction-computations-for-m1-to-m3.png)

So we have:

```
m[1] = 0.1g[1]
m_hat[1] = 1g[1]

m[2] = 0.09g[1] + 0.1g[2]
m_hat[2] = 0.474g1 + 0.526g2

m[3] = 0.081g1 + 0.09g2 + 0.1g3
m_hat[3] = 0.299g1 + 0.332g2 + 0.369g3
```

One key observation is that the weights now add up to 1, but does not force a 0.9-0.1 split.
Indeed, it doesn't make sense to force a 0.9-0.1 split early on when there aren't enough
past gradient values to distribute the 0.9 weight over.

Looking at the numbers makes me appreciate the properties of this approach, as well as the flaws in my
earlier attempt more clearly:

- m[1] has only gradient, it makes sense to use the full g1 value for this estimate since it's the only one available. It doesn't make sense to attenuate it.
- The key issue that the correction is trying to solve is that without it, we have unnecessarily small weights for the initial values.
- The correction also normalizes the weights to add up to 1 from the first iteration.
- The correction also ensures that the latest gradient has the largest individual weight
- The correction ensures that the weights decay as the gradients get older. (need to verify that decay is exponentia;)
- When we have sufficiently many gradients and beta > 0.5, the aggregate weights of the older gradients exceed the weight of the latest gradient (the number of gradients it takes to reach this depends on beta)
- The higher the iteration, the closer we approach to the 0.9 - 0.1 split between the older and current gradients, but
this split is not forced, it emerges gradually as we get more and more gradients to distribute the weights over.
- We create new variables for the bias-corrected estimates `m_hat[t]` and `v_hat[t]` instead of overwriting `m[t]` and `v[t]`. This means
the original values of `m[t]` and `v[t]` are preserved for the next iteration, making this approach suitable
for a recursive algorithm.


Out of curiosity, we can ask ourselves, how many iterations will it take for the gradients weights to reach a 0.9-0.1 split.

We know that in `m[t]`, the weight of the current gradient `g[t]` is `1-beta`, so in `m_hat[t]`, the weight will be adjusted
to `(1-beta)/(1-beta**t)`. So what we're asking as at what timestep `t` does this adjusted weight reach `0.1`. Well
it will never reach 0.1, but it will get infinitely closer to 0.1 as the weight of the latest gradient gets smaller with
each iteration.

![alt text](adam-equation-iterations-to-target-weight-of-latest-gradient-after-bias-correction.png)

So we use the formula:

```
t = ln(1 - (1-beta)/w) / ln(beta)
```

(Note that if `w == 1-beta`, we have the undefined `log(0)`, but the limit is infinity).

- For `beta = 0.9` and `w = 0.11`, `t = 23 (approx)`
- For `beta = 0.999` and `w = 0.0011`, `t = 2397 (approx)`

Surprised to find that in the case of `0.9`, I get a figure that's close to what I had gotten when I was
checking how many steps it would take to reach 0.9-0.1 split using the unadjusted `m[t]` formula (not
sure if I still have that in this doc, I may have removed it). But the difference is much larger
when beta is 0.999, in my earlier calculations it was > 6k, so perhaps it's just a coincidence that
it maches closely for 0.9.

#### 4.3.1.e Reviewing the fatal flaws in my attempt to derive Adam's bias correction

In my approach, enforcing the 0.9-0.1 split causes a major issue. Let's say we have 3 gradients, and the latest gradient
is forced to have weight 0.1, if we distribute 0.9 over the remaining 2 then at least one of them must have a weight
greater than 0.1, i.e. greater than the weight of the current gradient. This breaks key properties and assumptions
of the algorithm, i.e. that latest gradient has the largest individual weight and that weights of older gradients
decay exponentially. I noticed this contradiction when thinking about my approach but at that point I was
mentally exhausted and stuck and didn't know what to do, which is one of the reasons I decided to look at the paper.
But it should have been evidence that trying to force 0.9-0.1 split was wrong from the start.

But after noticing that in Adam's bias correction approach, the weights add up to 1, I feel like my intuition
was probably not as far off as I thought. If I change my thinking from forcing a 0.9 weight for older
gradients, to forcing an total aggregate weight of 1, then maybe there's a shot. But conceptually,
why is forcing an aggregate weight of 1 a desirable thing?

Well at `m[1]`, we assign weight `0.1` to `g[1]`. But this small weight assumes that there many other
gradients which share the remaining `0.9`. But at the beginning there are no other gradients due to
the zero initialization. So we unnecessarily penalize `g[1]`. And in subsequent iterations,
the few existing gradients continue to get penalized, contributing a small aggregate weight
even though there are no other gradients to consume the "unallocated" weight. The spirit of the
correction is to allocate the "entire" weight to the existing gradients without leaving any
"unallocated" weight to non-existing gradients. So if we allocate all the weight to existing
gradients, then the weights should sum up to 1.

We also want to preserve the other properties,
i.e. that the latest gradient has the individual largest weight, that weights decay
as gradients get older and when we have sufficient gradients, the aggregate
weight of previous gradients will exceed that of the latest. The original `m[t]` already
bears some of these properties:
- The latest gradient has weight `1-beta` which is always the largest individual, since all gradients start with 1-beta and get progressively attenuated in each iteraton by being multiplied by beta which is < 1.

We already calculated the total weight of all gradients: `1 - beta**t`. In my earlier
attempt I had asked the question, what factor can I mulitply the previous weights by to get `1-beta`.
Now I can ask a simpler question, what factor can I multiply the total aggregate weights by to get `1`.

```
f * (1 - beta ** t) = 1
f = 1 / (1 - beta ** t)
```

So I just divide by `(1 - beta ** t)`?

So if we divide all gradients by (1 - beta ** t), the aggregate weight will be 1, what about the other properties:

- If all weights add up to 1, and the weight of the latest gradient approaches (1 - beta), then after sufficient iterations,
the remaining weights will have aggregate weight close to beta, which is greater than (1- beta).

Will this matches the solution appears to match the from the paper, I think the justification is not as robust as the
statistical grounding of the paper. But it was still interesting to see if an adjustment of my earlier reasoning
would have led me to the same conclusion, even if with weaker mathematical support.
