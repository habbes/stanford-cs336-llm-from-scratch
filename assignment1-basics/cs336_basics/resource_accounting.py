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
    batch_size: int

    @property
    def d_k(self):
        return self.d_model // self.num_heads

class ResourceCounter(ABC):
    def __init__(self, name: str):
        self.name = name

    def get_name(self):
        return self.name

    @abstractmethod
    def get_resource_count(self, hyper_params: HyperParams):
        pass

class LeafCounter(ResourceCounter):
    def __init__(self, name: str, params_fn: Callable[[HyperParams], int]):
        super().__init__(name)
        self.params_fn = params_fn
    
    def get_resource_count(self, hyper_params: HyperParams):
        return self.params_fn(hyper_params)

class CompositeCounter(ResourceCounter):
    def __init__(self, name: str, *children: ResourceCounter):
        super().__init__(name)
        self.children = children
    
    def get_resource_count(self, hyper_params: HyperParams):
        return sum(
            map(lambda child: child.get_resource_count(hyper_params), self.children))

class RepeatCounter(ResourceCounter):
    def __init__(self, name: str, get_num_elements: Callable[[HyperParams], int], element: ResourceCounter):
        super().__init__(name)
        self.get_num_elements = get_num_elements
        self.element = element
    
    def get_resource_count(self, hyper_params: HyperParams):
        return self.get_num_elements(hyper_params) * self.element.get_resource_count(hyper_params)


def format_count(count: int, parent_count: int, total_count: int):
    pct_parent = 100 * count / parent_count
    pct_total = 100 * count / total_count
    return f"{count} ({pct_parent:.2f}% of parent) ({pct_total:.2f}% of total)"

def print_component_resource_counts(
        counter: ResourceCounter,
        hyper_params: HyperParams,
        indent_level: int = 0,
        parent_count: int = -1,
        total_count: int = -1):
    INDENT_SIZE = 2
    indent = " " * (indent_level * INDENT_SIZE) if indent_level > 0 else ""

    total_count = counter.get_resource_count(hyper_params) if total_count == -1 else total_count
    parent_count = total_count if parent_count == -1 else parent_count

    if isinstance(counter, RepeatCounter):
        count = counter.get_resource_count(hyper_params)
        print(f"{indent}{counter.name} x{counter.get_num_elements(hyper_params)}: {format_count(count, parent_count, total_count)}")
        print_component_resource_counts(counter.element, hyper_params, indent_level + 1, count, total_count)

    elif isinstance(counter, CompositeCounter):
        count = counter.get_resource_count(hyper_params)
        print(f"{indent}{counter.name}: {format_count(count, parent_count, total_count)}")
        for child in counter.children:
            print_component_resource_counts(child, hyper_params, indent_level + 1, count, total_count)

    else:
        print(f"{indent}{counter.name}: {format_count(counter.get_resource_count(hyper_params), parent_count, total_count)}")


def create_transformer_params_counter():
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
        RepeatCounter("TransformerLayers", lambda h: h.num_layers,
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

def create_transformer_flops_counter():
    """
    Creates a resource counter that estimates the number
    of FLOPs in a transformer architecture forward training pass
    based on the matrix multiply operations in the architecture's
    components.
    """
    flops_counter = CompositeCounter("TransformerLM",
        RepeatCounter("TransformerLayers", lambda h: h.num_layers,
            CompositeCounter("TransformerBlock",
                CompositeCounter("MultiHeadSelfAttention",
                    LeafCounter("Wq(x)", lambda h: 2 * h.batch_size * h.context_length * h.d_model * h.num_heads * h.d_k),
                    LeafCounter("Wk(x)", lambda h: 2 * h.batch_size * h.context_length * h.d_model * h.num_heads * h.d_k),
                    LeafCounter("Wv(x)", lambda h: 2 * h.batch_size * h.context_length * h.d_model * h.num_heads * h.d_k),
                    LeafCounter("RoPE(Q)", lambda h: 2 * h.batch_size * h.num_heads * h.context_length * h.d_k),
                    LeafCounter("RoPE(K)", lambda h: 2 * h.batch_size * h.num_heads * h.context_length * h.d_k),
                    CompositeCounter("ScaledDotProductAttention",
                        LeafCounter("Q @ K.T", lambda h: 2 * h.batch_size * h.num_heads * h.context_length * h.d_k * h.context_length),
                        LeafCounter("weights @ V", lambda h: 2 * h.batch_size * h.num_heads * h.context_length * h.context_length * h.d_k)),
                    LeafCounter("Wo(y)", lambda h: 2 * h.batch_size * h.num_heads * h.context_length * h.context_length * h.d_k)),
                CompositeCounter("FFSwiGLU",
                    LeafCounter("W1(x)", lambda h: 2 * h.batch_size * h.context_length * h.d_model * h.d_ff),
                    LeafCounter("W3(x)", lambda h: 2 * h.batch_size * h.context_length * h.d_model * h.d_ff),
                    LeafCounter("W2(x)", lambda h: 2 * h.context_length * h.d_ff * h.d_model)))),
        LeafCounter("Linear (LM Head)", lambda h: 2 * h.batch_size * h.context_length * h.d_model * h.vocab_size))
    
    return flops_counter


def get_gpt2_xl_config():
    return HyperParams(
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=4288, # (the nearest multiple of 64 to (8/3) * 1600)
        batch_size=1
    )

def get_gpt2_small_config():
    config = get_gpt2_xl_config()
    config.num_layers = 12
    config.d_model = 768
    config.d_ff = 2048 # (8/3) * 768
    config.num_heads = 12
    return config

def get_gpt2_medium_config():
    config = get_gpt2_xl_config()
    config.num_layers = 24
    config.d_model = 1024
    config.d_ff = 2752 # nearest multiple of 64 to (8/3) * 1024
    config.num_heads = 16
    return config

def get_gpt2_large_config():
    config = get_gpt2_xl_config()
    config.num_layers = 32
    config.d_model = 1280
    config.d_ff = 3456 # nearest multiple of 64 to (8/3) * 1280
    config.num_heads = 20
    return config

def gpt2_xl_trainable_params():
    gpt2_xl_config = get_gpt2_xl_config()
    gpt2_xl_params_counter = create_transformer_params_counter()
    param_count = gpt2_xl_params_counter.get_resource_count(gpt2_xl_config)

    memory_required = param_count * 4 # Assumes each param is float32
    memory_in_gigs = memory_required / 2**30

    print(f"GPT2 XL architecture has {param_count} trainable params and requires {memory_required} bytes, {memory_in_gigs} GiB")


def print_forward_pass_flops_breakdown(name: str, config: HyperParams):
    counter = create_transformer_flops_counter()
    flops = counter.get_resource_count(config)

    print(f"{name} architecture requires {flops} FLOPs for matrix multiplies in the forward pass.")
    print(f"{name} component FLOPs breakdown:")
    print_component_resource_counts(counter, config)

def gpt2_xl_forward_pass_flops():
    config = get_gpt2_xl_config()
    print_forward_pass_flops_breakdown("GPT2 XL", config)

def gpt2_small_forward_pass_flops():
    config = get_gpt2_small_config()
    print_forward_pass_flops_breakdown("GPT2 Small", config)

def gpt2_medium_forward_pass_flops():
    config = get_gpt2_medium_config()
    print_forward_pass_flops_breakdown("GPT2 Medium", config)

def gpt2_large_forward_pass_flops():
    config = get_gpt2_large_config()
    print_forward_pass_flops_breakdown("GPT2 Large", config)

def gpt2_xl_increased_context_length_forward_pass_flops():
    config = get_gpt2_xl_config()
    config.context_length = 16_384
    print_forward_pass_flops_breakdown("GPT2 XL (context length = 16,384)", config)


if __name__ == '__main__':
    gpt2_xl_trainable_params()
    print()
    gpt2_xl_forward_pass_flops()
    print()
    print()
    gpt2_small_forward_pass_flops()
    print()
    print()
    gpt2_medium_forward_pass_flops()
    print()
    print()
    gpt2_large_forward_pass_flops()
    print()
    print()
    gpt2_xl_increased_context_length_forward_pass_flops()

