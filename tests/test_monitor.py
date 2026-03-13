"""Unit tests for the DefragMonitor."""

import torch
import pytest
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpudefrag.monitor import DefragMonitor
from gpudefrag.utils import DefragConfig


class TestDefragMonitor:
    def test_start_stop(self):
        monitor = DefragMonitor(threshold=0.9)
        monitor.start()
        time.sleep(0.2)
        monitor.stop()
        stats = monitor.stats()
        assert "total_predictions" in stats
        assert "total_compactions" in stats

    def test_record_alloc(self):
        monitor = DefragMonitor(threshold=0.9)
        monitor.record_alloc(1024 * 1024, is_alloc=True)
        monitor.record_alloc(512 * 1024, is_alloc=False)
        # Should not crash

    def test_stats_structure(self):
        monitor = DefragMonitor()
        stats = monitor.stats()
        required_keys = ["total_predictions", "total_compactions", "total_freed_mb",
                         "avg_prediction_score", "killed"]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
