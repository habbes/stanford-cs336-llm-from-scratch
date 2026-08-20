import torch
from .optimizers import SGD

def training_loop(lr=1, iters=100):
    # Typical training loop structure:
    # each iteration computes loss and run a step of the optimizer.

    # When training actual models, learnable parameters will come from the model (m.parameters())
    # and the loss will be computed over a sample batch of data.

    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    for t in range(iters):
        opt.zero_grad() # Reset gradients for all learnable params
        loss = (weights ** 2).mean() # Compute scalar loss value
        print(loss.cpu().item())

        loss.backward() # Run backward pass which computes gradients
        opt.step() # Run optimizer step (updates parameters).


def run_training_loop(lr, iters):
    print(f"Running training loop with LR={lr} and num iterations={iters}")
    training_loop(lr=lr, iters=iters)
    print("Done.")

if __name__ == "__main__":
    run_training_loop(lr=1, iters=10)
    print()
    run_training_loop(lr=1e1, iters=10)
    print()
    run_training_loop(lr=1e2, iters=10)
    print()
    run_training_loop(lr=1e3, iters=10)


