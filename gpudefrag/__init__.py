"""
gpudefrag — Predictive GPU Memory Defragmenter
================================================

A Transformer-driven proactive CUDA memory optimizer for PyTorch.

Quick Start::

    from gpudefrag import auto_instrument

    # Wrap your model and optimizer with zero code changes
    model, optimizer = auto_instrument(model, optimizer)

    # ... standard training loop ...
"""

__version__ = "2.0.0"
__author__ = "GPU Defrag Infrastructure Team"

from gpudefrag.scheduler.monitor import DefragMonitor
from gpudefrag.trainer.callback import DefragCallback
from gpudefrag.trainer.auto_instrument import auto_instrument
from gpudefrag.trainer.ddp import DDPSyncManager
from gpudefrag.profiler.collector import AllocationCollector
from gpudefrag.scheduler.predictor import FragPredictor
from gpudefrag.defrag_engine.defragmenter import GPUMemoryDefragmenter

# Re-exported from migrated modules for unified namespace
from gpudefrag.profiler.allocator_logger import AllocatorLogger
from gpudefrag.scheduler.risk_model import OOMRiskModel
from gpudefrag.trainer.training_hook import TrainingHook
from gpudefrag.defrag_engine.policy import MitigationPolicy

__all__ = [
    "DefragMonitor",
    "DefragCallback",
    "auto_instrument",
    "DDPSyncManager",
    "AllocationCollector",
    "FragPredictor",
    "GPUMemoryDefragmenter",
    "AllocatorLogger",
    "OOMRiskModel",
    "TrainingHook",
    "MitigationPolicy",
]
