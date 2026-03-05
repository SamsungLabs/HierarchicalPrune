"""Forward hooks for capturing intermediate feature activations during distillation."""


def get_activation(mem, name):
    """Create a forward hook that stores a module's output in ``mem[name]``.

    Args:
        mem: Dictionary to store captured activations.
        name: Key under which the activation will be stored.

    Returns:
        A hook function compatible with ``register_forward_hook``.
    """

    def get_output_hook(module, input, output):
        mem[name] = output

    return get_output_hook


def add_hook(net, mem, mapping_layers):
    """Register forward hooks on named modules for feature-level knowledge distillation.

    Args:
        net: The model to attach hooks to.
        mem: Dictionary where captured activations will be stored.
        mapping_layers: List of module names.
    """
    for n, m in net.named_modules():
        if n in mapping_layers:
            m.register_forward_hook(get_activation(mem, n))
