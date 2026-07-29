from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Collection

@dataclass
class HyperParams:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int

    @property
    def d_k(self):
        return self.d_model // self.num_heads

class ParamCounter(ABC):
    def __init__(self, name: str):
        self.name = name

    def get_name(self):
        return self.name

    @abstractmethod
    def get_num_params(self, hyper_params: HyperParams):
        pass

class LeafCounter(ParamCounter):
    def __init__(self, name: str, params_fn: Callable[[HyperParams], int]):
        super().__init__(name)
        self.params_fn = params_fn
    
    def get_num_params(self, hyper_params: HyperParams):
        return self.params_fn(hyper_params)

class CompositeCounter(ParamCounter):
    def __init__(self, name: str, *children: ParamCounter):
        super().__init__(name)
        self.children = children
    
    def get_num_params(self, hyper_params: HyperParams):
        return sum(
            map(lambda child: child.get_num_params(hyper_params), self.children))


def create_params_counter(num_layers: int):
    params_counter = CompositeCounter("TransformerLM",
        LeafCounter("Embedding", lambda h: h.vocab_size * h.d_model),
        # TODO: is RoPE worth mentioning, it has no trainable params but holds some buffer
        # LeafContainer("RoPE", lambda h: h.context_length * h.d_k * h.d_k),
        CompositeCounter("RMSNorm",
            LeafCounter("g", lambda h: h.d_model),
        CompositeCounter("TransformerLayers",
            (CompositeCounter("TransformerBlock",
                CompositeCounter("RMSNorm",
                    LeafCounter("g", lambda h: h.d_model),
                CompositeCounter("MultiHeadSelfAttention",
                    LeafCounter("Wq", lambda h: h.d_model * h.num_heads * h.d_k)))) for _ in range(num_layers)))),
                    # TODO complete MHSA and TransformerBlock
        CompositeCounter("RMSNorm (output)",
            LeafCounter("g", lambda h: h.d_model)),
        LeafCounter("Linear (LM Head)", lambda h: h.d_model * h.vocab_size))
    
    return params_counter
