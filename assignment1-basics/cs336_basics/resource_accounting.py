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

class ParamContainer(ABC):
    def __init__(self, name: str):
        self.name = name

    def get_name(self):
        return self.name

    @abstractmethod
    def get_num_params(self, hyper_params: HyperParams):
        pass

class LeafContainer(ParamContainer):
    def __init__(self, name: str, params_fn: Callable[[HyperParams], int]):
        super().__init__(name)
        self.params_fn = params_fn
    
    def get_num_params(self, hyper_params: HyperParams):
        return self.params_fn(hyper_params)

class ParentContainer(ParamContainer):
    def __init__(self, name: str, *children: ParamContainer):
        super().__init__(name)
        self.children = children
    
    def get_num_params(self, hyper_params: HyperParams):
        return sum(
            map(lambda child: child.get_num_params(hyper_params), self.children))


def create_params_counter(num_layers: int):
    params_counter = ParentContainer("TransformerLM",
        LeafContainer("Embedding", lambda h: h.vocab_size * h.d_model),
        # TODO: is RoPE worth mentioning, it has no trainable params but holds some buffer
        # LeafContainer("RoPE", lambda h: h.context_length * h.d_k * h.d_k),
        ParentContainer("RMSNorm",
            LeafContainer("g", lambda h: h.d_model),
        ParentContainer("TransformerLayers",
            (ParentContainer("TransformerBlock",
                ParentContainer("RMSNorm",
                    LeafContainer("g", lambda h: h.d_model),
                ParentContainer("MultiHeadSelfAttention",
                    LeafContainer("Wq", lambda h: h.d_model * h.num_heads * h.d_model)))) for _ in range(num_layers)))),
                    # TODO complete MHSA and TransformerBlock
        ParentContainer("RMSNorm (output)",
            LeafContainer("g", lambda h: h.d_model)),
        LeafContainer("Linear (LM Head)", lambda h: h.d_model * h.vocab_size))
    
    return params_counter
