import torch
import math
from torch import nn

from collections.abc import Callable, Iterable
from typing import Optional

from tests import adapters

class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] 
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss

class AdamW(torch.optim.Optimizer):

    def __init__(self, 
                 params,
                 lr: float,
                 weight_decay: float,
                 betas: tuple[float, float],
                 eps: float
                 ):
         defaults = {
             "alpha": lr, "beta_1": betas[0], "beta_2": betas[1], "weight_decay": weight_decay, "eps": eps
             }
         super().__init__(params, defaults)

    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            wd = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # initialize state on first iter
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros(p.shape, device=p.device, dtype=p.dtype)
                    state["v"] = torch.zeros(p.shape, device=p.device, dtype=p.dtype)

                m = state["m"]
                v = state["v"]
                t = state["t"] 

                grad = p.grad.data # get the gradient loss

                # update first, second moment est. (in place)
                m.mul_(beta_1).add_(grad, alpha=1-beta_1)
                v.mul_(beta_2).addcmul_(grad, grad, value=1-beta_2)

                # compute adjusted alpha 
                alpha_t = alpha * math.sqrt(1 - (beta_2) ** t) / (1 - (beta_1) ** t)
            
                # update parameters w/ weight decay (in place)
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-alpha_t)
                p.data.add_(p.data, alpha=-alpha*wd)

                state["t"] += 1 # increment iteration num

        return loss




        


