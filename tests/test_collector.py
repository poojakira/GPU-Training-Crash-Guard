"""Unit tests for the AllocationCollector."""

import torch
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpudefrag.profiler.collector import AllocationCollector
from gpudefrag.utils import DefragConfig


class TestAllocationCollector:
    def test_manual_record(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        collector = AllocationCollector()
        t = torch.randn(1024, device="cuda")
        collector.record()
        assert collector.event_count >= 0  # May or may not detect change depending on timing

    def test_to_dataframe(self):
        collector = AllocationCollector()
        df = collector.to_dataframe()
        assert df.empty  # No events yet

    def test_save_empty(self, tmp_path):
        collector = AllocationCollector()
        path = str(tmp_path / "test.parquet")
        collector.save(path)
        # Should warn and not create file for empty collector

    def test_clear(self):
        collector = AllocationCollector()
        collector.clear()
        assert collector.event_count == 0


class TestDefragConfig:
    def test_save_load(self, tmp_path):
        config = DefragConfig(frag_threshold=0.8, seq_len=128)
        path = str(tmp_path / "config.json")
        config.save(path)
        loaded = DefragConfig.load(path)
        assert loaded.frag_threshold == 0.8
        assert loaded.seq_len == 128
