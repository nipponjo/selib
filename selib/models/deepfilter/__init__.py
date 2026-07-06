"""DeepFilterNet ONNX runtime wrappers.

The public entry point is ``DeepFilterNetOnnx``. It loads Option-B ONNX exports
created from DeepFilterNet models and runs enhancement with NumPy,
ONNX Runtime, and the pure-Python DeepFilterNet frontend/backend in this
package.
"""

from .runtime import DeepFilterNetOnnx

__all__ = ["DeepFilterNetOnnx"]
