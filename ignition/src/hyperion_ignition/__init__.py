"""hyperion-ignition — build-time compiler for .hyperion packages.

Phase 1 of the two-stage architecture. This stays Python (per the hybrid
strategy) to reuse Hugging Face / PyTorch / TensorRT tooling. The module
layout is plugin-shaped — adding a model architecture or a hardware recipe is
a new file, never an edit to existing ones — so the long-term Rust port can
lift each submodule independently without touching helios/.
"""

__version__ = "0.1.0"
