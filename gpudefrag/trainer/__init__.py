"""gpudefrag.trainer — Training pipeline, auto-instrumentation, and DDP support."""

from gpudefrag.trainer.callback import DefragCallback
from gpudefrag.trainer.auto_instrument import auto_instrument
from gpudefrag.trainer.ddp import DDPSyncManager
from gpudefrag.trainer.training_hook import TrainingHook

__all__ = ["DefragCallback", "auto_instrument", "DDPSyncManager", "TrainingHook"]
