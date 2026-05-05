"""
Persistence module - minimal implementation for ImagenTime U-Net compatibility.

The official ImagenTime code uses @persistence.persistent_class decorator.
This is a simple implementation that just returns the class unchanged.
"""

import functools


def persistent_class(cls):
    """
    Decorator for persistent classes - ImagenTime compatibility.
    In the original NVIDIA code, this handles model state persistence.
    Here we use it as an identity decorator for compatibility.
    
    Args:
        cls: Class to decorate
    
    Returns:
        Unchanged class
    """
    return cls
