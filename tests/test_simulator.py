"""
tests/test_simulator.py — Verify the workload simulator produces correct data.
"""

import numpy as np

from scripts.workload_simulator import (
    GPUWorkload,
    TransformerSpec,
    CNNSpec,
    CachingAllocator,
    _quantize,
    BLOCK_SIZE_MB,
)


class TestCachingAllocator:
    """Tests for the block-level allocator model."""

    def test_alloc_increases_memory(self):
        alloc = CachingAllocator(vram_mb=1024)
        bid = alloc.alloc(10.0, "test", 0)
        assert bid is not None
        assert alloc.allocated_mb > 0

    def test_free_decreases_allocated(self):
        alloc = CachingAllocator(vram_mb=1024)
        bid = alloc.alloc(10.0, "test", 0)
        before = alloc.allocated_mb
        alloc.free(bid)
        assert alloc.allocated_mb < before

    def test_free_caches_block(self):
        alloc = CachingAllocator(vram_mb=1024)
        bid = alloc.alloc(10.0, "test", 0)
        alloc.free(bid)
        # Reserved stays the same (cached), but allocated drops
        assert alloc.reserved_mb > 0
        assert alloc.free_cached_mb > 0

    def test_empty_cache_releases_reserved(self):
        alloc = CachingAllocator(vram_mb=1024)
        bid = alloc.alloc(50.0, "test", 0)
        alloc.free(bid)
        reserved_before = alloc.reserved_mb
        alloc.empty_cache()
        assert alloc.reserved_mb < reserved_before
        assert alloc.free_cached_mb == 0

    def test_oom_returns_none(self):
        alloc = CachingAllocator(vram_mb=100)
        bid = alloc.alloc(200.0, "test", 0)
        assert bid is None

    def test_fragmentation_range(self):
        alloc = CachingAllocator(vram_mb=1024)
        alloc.alloc(100, "a", 0)
        alloc.alloc(100, "b", 0)
        frag = alloc.fragmentation
        assert 0.0 <= frag <= 1.0

    def test_quantize(self):
        assert _quantize(0.5) == BLOCK_SIZE_MB
        assert _quantize(2.0) == BLOCK_SIZE_MB
        assert _quantize(3.0) == 2 * BLOCK_SIZE_MB
        assert _quantize(10.0) == 10  # 10 / 2 = 5 blocks


class TestTransformerSpec:
    """Tests for architecture specifications."""

    def test_gpt2_has_positive_sizes(self):
        spec = TransformerSpec.gpt2()
        assert spec.param_mb > 0
        assert spec.activation_per_layer_mb > 0
        assert spec.gradient_mb > 0
        assert spec.optimizer_state_mb > 0

    def test_larger_model_has_more_params(self):
        small = TransformerSpec.gpt2()
        large = TransformerSpec.gpt2_medium()
        assert large.param_mb > small.param_mb

    def test_larger_batch_more_activations(self):
        small_bs = TransformerSpec.gpt2(batch_size=2)
        large_bs = TransformerSpec.gpt2(batch_size=16)
        assert large_bs.activation_per_layer_mb > small_bs.activation_per_layer_mb

    def test_longer_seq_quadratic_activations(self):
        short = TransformerSpec.gpt2(seq_len=128)
        long = TransformerSpec.gpt2(seq_len=512)
        # Attention is quadratic in seq_len, so 4x seq → ~16x attention
        ratio = long.activation_per_layer_mb / short.activation_per_layer_mb
        assert ratio >= 8  # Should be significantly more than 4x

    def test_optimizer_is_2x_params(self):
        spec = TransformerSpec.gpt2()
        assert abs(spec.optimizer_state_mb - 2 * spec.param_mb) < 0.01


class TestCNNSpec:
    def test_resnet50_has_positive_sizes(self):
        spec = CNNSpec.resnet50()
        assert spec.param_mb > 0
        assert spec.activation_per_layer_mb > 0


class TestGPUWorkload:
    """Tests for the full workload simulator."""

    def test_produces_events(self):
        wl = GPUWorkload(TransformerSpec.gpt2(), vram_mb=8192)
        events = wl.run(steps=10, seed=42)
        assert len(events) > 50  # Multiple events per step

    def test_events_have_required_fields(self):
        wl = GPUWorkload(TransformerSpec.gpt2(), vram_mb=8192)
        events = wl.run(steps=5, seed=42)
        required = {"timestamp_ns", "step", "phase", "action", "delta_bytes",
                     "abs_allocated", "abs_reserved", "fragmentation",
                     "utilization", "tag", "oom"}
        for e in events:
            assert required.issubset(e.keys()), f"Missing fields: {required - set(e.keys())}"

    def test_fragmentation_is_nontrivial(self):
        """The whole point: simulator must produce real fragmentation."""
        wl = GPUWorkload(TransformerSpec.gpt2(), vram_mb=8192)
        events = wl.run(steps=100, seed=42)
        frags = [e["fragmentation"] for e in events]
        assert max(frags) > 0.05, f"Max frag {max(frags)} is too low"
        assert np.std(frags) > 0.01, f"Frag std {np.std(frags)} too low — no variance"

    def test_phases_present(self):
        wl = GPUWorkload(TransformerSpec.gpt2(), vram_mb=8192)
        events = wl.run(steps=20, seed=42)
        phases = set(e["phase"] for e in events)
        assert "forward" in phases
        assert "backward" in phases
        assert "optimizer" in phases

    def test_oom_under_pressure(self):
        """Tight VRAM should produce OOM events."""
        wl = GPUWorkload(TransformerSpec.gpt2(batch_size=16), vram_mb=512)
        events = wl.run(steps=50, seed=42)
        ooms = sum(1 for e in events if e["oom"])
        assert ooms > 0, "Expected OOMs under extreme memory pressure"

    def test_no_oom_with_headroom(self):
        """Plenty of VRAM should produce zero OOMs."""
        wl = GPUWorkload(TransformerSpec.gpt2(batch_size=2, seq_len=128), vram_mb=32768)
        events = wl.run(steps=50, seed=42)
        ooms = sum(1 for e in events if e["oom"])
        assert ooms == 0, f"Got {ooms} OOMs with 32 GB VRAM"

    def test_cnn_workload(self):
        wl = GPUWorkload(CNNSpec.resnet50(batch_size=8), vram_mb=8192)
        events = wl.run(steps=20, seed=42)
        assert len(events) > 20

    def test_deterministic_with_seed(self):
        wl1 = GPUWorkload(TransformerSpec.gpt2(), vram_mb=8192)
        wl2 = GPUWorkload(TransformerSpec.gpt2(), vram_mb=8192)
        e1 = wl1.run(steps=10, seed=123)
        e2 = wl2.run(steps=10, seed=123)
        # Same seed → same number of events and same alloc patterns
        assert len(e1) == len(e2)
        # Tags and actions should match (fragmentation may differ slightly
        # due to timing-dependent rounding)
        tags1 = [(e["step"], e["action"], e["tag"]) for e in e1]
        tags2 = [(e["step"], e["action"], e["tag"]) for e in e2]
        assert tags1 == tags2
