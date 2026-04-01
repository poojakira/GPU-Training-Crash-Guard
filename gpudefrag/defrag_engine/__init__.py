"""gpudefrag.defrag_engine — GPU memory defragmentation engine."""

from gpudefrag.defrag_engine.defragmenter import GPUMemoryDefragmenter
from gpudefrag.defrag_engine.compactor import MemoryCompactor
from gpudefrag.defrag_engine.policy import MitigationPolicy

__all__ = ["GPUMemoryDefragmenter", "MemoryCompactor", "MitigationPolicy"]
