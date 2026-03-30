"""
gpudefrag — Predictive GPU Memory Defragmenter
================================================

A Transformer-driven proactive CUDA memory optimizer for PyTorch.

Quick Start::

    from gpudefrag import DefragMonitor, DefragCallback

    # Option 1: Background monitor
    monitor = DefragMonitor(threshold=0.7)
    monitor.start()

    # Option 2: Training callback
    callback = DefragCallback()
"""

__version__ = "1.0.0"
__author__ = "GPU Defrag Team"

from gpudefrag.scheduler.monitor import DefragMonitor
from gpudefrag.trainer.callback import DefragCallback
from gpudefrag.profiler.collector import AllocationCollector
from gpudefrag.scheduler.predictor import FragPredictor
from gpudefrag.defrag_engine.compactor import MemoryCompactor

__all__ = [
    "DefragMonitor",
    "DefragCallback",
    "AllocationCollector",
    "FragPredictor",
    "MemoryCompactor",
]
