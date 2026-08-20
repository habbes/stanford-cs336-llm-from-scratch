from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    """
    A modified version of SGD (stochastic gradient descent)
    where the learning rate decays over time.
    The parameter update uses the following formula:
    p_next = p - (lr/sqrt(t + 1)) * gradient(loss(parameters, batch))

    where p is the value of a parameter at time t and p_next
    the value of the same parameter at time t + 1.
    """
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}.")
        
        # We must pass the parameters and default hyperparameters
        # to the base constructor.
        # The parameters might come in groups, but if we pass
        # a single collection of parameter objects, then the
        # base constructor will create a single group and assign
        # it the default hyperparams.
        defaults = { "lr": lr }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        # The step() method should update the model's parameters.
        # During a training loop, this is called after the backward() pass
        # and therefore has access to the gradients on the last batch.

        # The API specifies that the called might pass a closure
        # to recompute the loss. We add support for that to comply with the API
        # but we won't need it in our example.
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # We can use the parameter's state to store arbitrary data
                # related to the parameter during backprop, the state persists
                # across calls to the step() function.
                state = self.state[p] # Get the state associated with parameter p.
                t = state.get("t", 0) # Get the iteration number from the state, or 0 if it's not yet set.
                grad = p.grad.data # Get the gradient of the loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place
                state["t"] = t + 1 # Increment iteration number
        
        return loss