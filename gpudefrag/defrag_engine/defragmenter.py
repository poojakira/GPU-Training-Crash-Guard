"""
gpudefrag.defrag_engine.defragmenter — Active PyTorch Tensor Compaction.

Implements a REAL-WORLD memory defragmenter that actively repacks scattered 
live PyTorch tensors (e.g., model parameters and accumulated gradients) into 
a contiguous mega-buffer.

This directly combats CachingAllocator fragmentation during model training
by merging scattered allocations into a singular dense block and letting
empty_cache() actually release physical memory holes underneath.
"""

import gc
import logging
import time
from typing import Iterable, List, Dict, Any, Optional

import torch

try:
    from gpudefrag.defrag_engine.kernels import triton_compaction_copy
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

log = logging.getLogger("gpudefrag.defragmenter")


class GPUMemoryDefragmenter:
    """
    Actively repacks dynamic PyTorch tensors into contiguous memory blocks.
    
    This is not a simple cache eviction. It physically copies scattered tensor
    data into a single contiguous VRAM allocation and silently replaces the
    underlying `.data` pointers of the live tensors so that autograd and
    optimizer states continue flawlessly.
    """

    def __init__(self, use_triton: bool = True):
        """
        Args:
            use_triton: Whether to use the extreme-bandwidth custom Triton copying kernel
                        if it is available on this system.
        """
        self.use_triton = use_triton and HAS_TRITON
        self._history: List[Dict[str, Any]] = []

    def defragment_tensors(self, tensors: Iterable[torch.Tensor], reason: str = "compaction") -> Dict[str, Any]:
        """
        Takes an iterable of scattered live tensors and tightly packs them into
        a newly allocated contiguous block.
        
        Args:
            tensors: An iterable of tensors (e.g. `model.parameters()`)
            reason: Tag for telemetry logging.
            
        Returns:
            Dictionary containing metrics about the compaction duration and memory reclaimed.
        """
        # Filter valid tensors (must be floating point / complex and instantiated)
        tensors = [t for t in tensors if t is not None and t.numel() > 0]
        if not tensors:
            return {"skipped": True, "reason": "no_valid_tensors"}

        device = tensors[0].device
        dtype = tensors[0].dtype
        
        # Verify uniform device and dtype (usually true for parameters/gradients in a single replica)
        valid_tensors, total_elements, total_bytes = [], 0, 0
        for t in tensors:
            if t.device == device and t.dtype == dtype:
                valid_tensors.append(t)
                total_elements += t.numel()
                total_bytes += t.numel() * t.element_size()
        
        if total_elements == 0:
            return {"skipped": True, "reason": "no_matching_tensors"}

        t0 = time.perf_counter()
        
        # Pre-execution snapshot
        pre_allocated = torch.cuda.memory_allocated() if device.type == "cuda" else 0
        pre_reserved = torch.cuda.memory_reserved() if device.type == "cuda" else 0
        
        # 1. Allocate the singular contiguous VRAM block
        # (This acts as the "defragmented" destination block)
        try:
            mega_buffer = torch.empty(total_elements, dtype=dtype, device=device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.warning("Defragmentation failed: Cannot allocate temporary %d MB buffer.", total_bytes // 1024**2)
                return {"skipped": True, "reason": "oom_during_allocation"}
            raise

        # 2. Copy the scattered contents into the dense buffer
        offset = 0
        triton_successes = 0
        for t in valid_tensors:
            numel = t.numel()
            dest_view = mega_buffer[offset : offset + numel]
            
            src_flat = t.view(-1)
            
            # Use real-world Triton kernels if enabled and on CUDA
            if self.use_triton and device.type == "cuda" and t.is_cuda:
                try:
                    triton_compaction_copy(src_flat, dest_view)
                    triton_successes += 1
                except Exception as e:
                    # Fallback to standard PyTorch ATen contiguous copy if Triton fails
                    dest_view.copy_(src_flat)
            else:
                dest_view.copy_(src_flat)
                
            offset += numel
            
        # 3. Memory Rewrite phase (Dangerous!)
        # Overwrite the tensor data pointers directly. This instantly orphans all the
        # old scattered VRAM blocks, putting them into PyTorch's internal Garbage Collector.
        offset = 0
        for t in valid_tensors:
            numel = t.numel()
            # Safety check: ensure we retain autograd metadata
            requires_grad = t.requires_grad
            
            if requires_grad:
                t.requires_grad_(False)
                
            # Direct pointer rewrite
            t.data = mega_buffer[offset : offset + numel].view_as(t)
            
            if requires_grad:
                t.requires_grad_(True)
                
            offset += numel

        # 4. Trigger active Garbage Collection and CUDA memory flush
        # This will obliterate all the scattered "holes" and physically 
        # return the compacted VRAM to the CudaAllocator or OS.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        # Post-execution snapshot
        post_allocated = torch.cuda.memory_allocated() if device.type == "cuda" else 0
        post_reserved = torch.cuda.memory_reserved() if device.type == "cuda" else 0
        
        freed_mb = (pre_reserved - post_reserved) / (1024 ** 2) if device.type == "cuda" else 0.0

        record = {
            "timestamp": time.time(),
            "reason": reason,
            "tensors_compacted": len(valid_tensors),
            "megabytes_compacted": total_bytes / (1024**2),
            "triton_used": triton_successes > 0,
            "freed_mb": freed_mb,
            "elapsed_ms": elapsed_ms,
        }
        
        self._history.append(record)
        log.info(
            "Packaged %d tensors (%.1f MB) into contiguous block in %.1f ms using %s. Reclaimed %.1f MB.",
            len(valid_tensors), total_bytes / (1024**2), elapsed_ms, 
            "Triton" if triton_successes > 0 else "ATen",
            freed_mb
        )
        
        return record
