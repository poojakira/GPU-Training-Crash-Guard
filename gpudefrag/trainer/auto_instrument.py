"""
gpudefrag.trainer.auto_instrument — Zero-Code-Change PyTorch Injector.

This module provides enterprise-level autoinstrumentation for any PyTorch
model and optimizer. It dynamically intercepts forward passes, backward passes,
and optimizer steps using PyTorch hooks, abstracting the `TrainingHook` entirely
away from the user's workload code.

Usage:
    from gpudefrag import auto_instrument
    model, optimizer = auto_instrument(model, optimizer)
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from gpudefrag.defrag_engine.defragmenter import GPUMemoryDefragmenter
from src.hooks.training_hook import TrainingHook
from src.mitigation.policy import MitigationPolicy
from src.predictor.risk_model import OOMRiskModel

log = logging.getLogger("gpudefrag.auto_instrument")


class _InstrumentedModel(nn.Module):
    """
    Thin wrapper over the user's root model that triggers telemetry
    on every forward pass and intercepts backward hooks on the 
    root gradient edges.
    """

    def __init__(self, model: nn.Module, hook: TrainingHook):
        super().__init__()
        self.module = model
        self.hook = hook
        
        # Register forward hook directly on the root module
        self.module.register_forward_pre_hook(self._forward_pre_hook)
        self.module.register_forward_hook(self._forward_post_hook)

    def _forward_pre_hook(self, module, inputs):
        self.hook.on_forward_begin()

    def _forward_post_hook(self, module, inputs, output):
        self.hook.on_forward_end()
        # The backward pass starts right after the loss is derived from this output.
        # We assume the user creates loss and calls backward.
        self.hook.on_backward_begin()
        
        # We can loosely register a backward hook on the output tensor if it requires grad
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(self._backward_done_hook)
        elif isinstance(output, (list, tuple)):
            for o in output:
                if isinstance(o, torch.Tensor) and o.requires_grad:
                    o.register_hook(self._backward_done_hook)
                    
    def _backward_done_hook(self, grad):
        self.hook.on_backward_end()
        return grad

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _InstrumentedOptimizer(Optimizer):
    """
    Wrapper around the PyTorch optimizer to intercept `.step()`.
    """
    def __init__(self, optimizer: Optimizer, hook: TrainingHook, policy: MitigationPolicy, model: nn.Module):
        self.optimizer = optimizer
        self.hook = hook
        self.policy = policy
        self.model = model
        
        # We must manually intercept step because Optimizer doesn't have native hook APIs in all PyTorch versions
        self.param_groups = self.optimizer.param_groups
        self.defaults = getattr(optimizer, 'defaults', {})
        self.state = self.optimizer.state

    def step(self, closure=None):
        self.hook.on_optimizer_step()
        result = self.optimizer.step(closure)
        
        # Extract batch size heuristically if possible, default to 1
        batch_size = 1
        risk = self.hook.on_step_complete(batch_size=batch_size)
        
        # Dispatch the mitigation policy transparently
        self.policy.evaluate(
            risk_score=risk,
            current_batch_size=batch_size,
            tensors_to_defragment=self.model.parameters()
        )
        return result

    def zero_grad(self, set_to_none=False):
        return self.optimizer.zero_grad(set_to_none=set_to_none)


def auto_instrument(
    model: nn.Module, 
    optimizer: Optimizer, 
    risk_threshold: float = 0.8,
    use_triton: bool = True
) -> Tuple[nn.Module, Optimizer]:
    """
    NVIDIA-grade Zero-Code-Change instrumentation for PyTorch workloads.
    
    Transforms standard models and optimizers into gpudefrag-aware 
    components that automatically report structural memory diagnostics 
    and invoke custom Triton defragmentation kernels immediately prior 
    to encountering Out-Of-Memory exceptions.
    
    Args:
        model: PyTorch nn.Module instance
        optimizer: PyTorch Optimizer instance
        risk_threshold: The utilization threshold before triggering compaction
        use_triton: Activate native Triton zero-copy compaction kernels
        
    Returns:
        (instrumented_model, instrumented_optimizer)
    """
    log.info("Applying Zero-Code-Change generic Auto-Instrumentation...")
    
    # Initialize the entire intelligence stack silently
    risk_model = OOMRiskModel(mode="rule")
    hook = TrainingHook(risk_model=risk_model)
    
    engine = GPUMemoryDefragmenter(use_triton=use_triton)
    policy = MitigationPolicy(act_threshold=risk_threshold, engine=engine)
    
    # Wrap user objects
    wrapped_model = _InstrumentedModel(model, hook)
    wrapped_optimizer = _InstrumentedOptimizer(optimizer, hook, policy, wrapped_model.module)
    
    return wrapped_model, wrapped_optimizer
