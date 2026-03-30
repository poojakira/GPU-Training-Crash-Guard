"""
gpudefrag.optimization.quantization
===================================

Integrates memory-efficient numerical representation (quantization) pipelines
for PyTorch models, critical for large LLMs before running defragmentation loops.
"""

import torch
import torch.nn as nn
from gpudefrag.utils import get_logger

log = get_logger("quantization")

def apply_dynamic_quantization(model: nn.Module, dtype=torch.qint8) -> nn.Module:
    """
    Applies aggressive dynamic weight quantization to a PyTorch model.
    By compressing weight representations from fp32 -> int8, we free up massive
    blocks of memory for KV-caching during LLM inference, massively reducing the
    need for frequent defragmentation.
    """
    log.info(f"Applying CPU dynamic quantization to model logic using {dtype}")
    # We target nn.Linear as they account for >80% of parameters in Transformers
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=dtype
    )
    return quantized_model

def get_model_size_mb(model: nn.Module) -> float:
    """Calculates the physical memory footprint of a model."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / (1024**2)
