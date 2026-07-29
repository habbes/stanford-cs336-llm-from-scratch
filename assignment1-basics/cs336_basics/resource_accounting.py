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

class RepeatCounter(ParamCounter):
    def __init__(self, name: str, element_count: int, element: ParamCounter):
        super().__init__(name)
        self.element_count = element_count
        self.element = element
    
    def get_num_params(self, hyper_params: HyperParams):
        return self.element_count * self.element.get_num_params(hyper_params)


def create_transformer_params_counter(num_layers: int):
    """
    Creates a resource counter that returns the total
    number of trainable params in a transformer architecture
    with the specified number of transformer layers given
    a specific set of hyper parameters.
    """
    params_counter = CompositeCounter("TransformerLM",
        LeafCounter("Embedding", lambda h: h.vocab_size * h.d_model),
        # TODO: is RoPE worth mentioning, it has no trainable params but holds some buffer
        # LeafContainer("RoPE", lambda h: h.context_length * h.d_k * h.d_k),
        CompositeCounter("RMSNorm",
            LeafCounter("g", lambda h: h.d_model),
        RepeatCounter("TransformerLayers", num_layers,
            CompositeCounter("TransformerBlock",
                CompositeCounter("RMSNorm",
                    LeafCounter("g", lambda h: h.d_model),
                CompositeCounter("MultiHeadSelfAttention",
                    LeafCounter("Wq", lambda h: h.d_model * h.num_heads * h.d_k),
                    LeafCounter("Wk", lambda h: h.d_model * h.num_heads * h.d_k),
                    LeafCounter("Wv", lambda h: h.d_model * h.num_heads * h.d_k),
                    LeafCounter("Wo", lambda h: h.num_heads * h.d_k * h.d_model)),
                CompositeCounter("RMSNorm",
                    LeafCounter("g", lambda h: h.d_model)),
                CompositeCounter("FFSwiGLU",
                    LeafCounter("W1", lambda h: h.d_model * h.d_ff),
                    LeafCounter("W3", lambda h: h.d_model * h.d_ff),
                    LeafCounter("W2", lambda h: h.d_ff * h.d_model)),
                CompositeCounter("RMSNorm",
                    LeafCounter("g", lambda h: h.d_model))))),
        CompositeCounter("RMSNorm (output)",
            LeafCounter("g", lambda h: h.d_model)),
        LeafCounter("Linear (LM Head)", lambda h: h.d_model * h.vocab_size)))
    
    return params_counter


def get_gpt2_xl_config():
    return HyperParams(
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=4288 # (the nearest multiple of 64 to (8/3) * 1600)
    )

def gpt2_xl_trainable_params():
    gpt2_xl_config = get_gpt2_xl_config()
    gpt2_xl_params_counter = create_transformer_params_counter(gpt2_xl_config.num_layers)
    param_count = gpt2_xl_params_counter.get_num_params(gpt2_xl_config)

    memory_required = param_count * 4 # Assumes each param is float32
    memory_in_gigs = memory_required / 2**30

    print(f"GPT2 XL architecture has {param_count} trainable params and requires {memory_required} bytes, {memory_in_gigs} GiB")

if __name__ == '__main__':
    gpt2_xl_trainable_params()

