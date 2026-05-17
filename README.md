# RTX-OOM-Guard

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![CI](https://github.com/poojakira/RTX-OOM-Guard/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/RTX-OOM-Guard/actions)

**Proactive CUDA memory defragmenter for PyTorch that predicts and prevents GPU out-of-memory (OOM) crashes by actively compacting VRAM during training.**

---

## Problem

PyTorch's `CachingAllocator` leaves VRAM fragmented during long training runs, causing OOM crashes even when total free memory appears sufficient. Gradient checkpointing and reduced batch sizes sacrifice throughput. RTX-OOM-Guard predicts fragmentation before it causes a crash and proactively compacts live tensors into contiguous blocks.

---

## Key Features

- **`GPUMemoryDefragmenter`** — Repacks scattered model parameters into a contiguous VRAM buffer via `.data` pointer replacement. **Note:** optimizer state tensors (Adam's `exp_avg`, `exp_avg_sq`) and gradients are NOT migrated — they remain in their original allocations. This compacts parameters only.
- **Triton Copy Kernel** — Uses Triton `load`/`store` for the copy step when available; functionally equivalent to `tensor.copy_()` but exercises the Triton JIT path. Falls back to ATen otherwise.
- **`OOMRiskModel`** — Rule-based heuristic scoring OOM probability from memory utilization, fragmentation ratio, and allocation rate. No ML — just sigmoid-squashed weighted features.
- **`DefragMonitor`** — Background daemon polling at configurable intervals; triggers compaction when risk score exceeds threshold.
- **`AllocationCollector`** — Hooks into PyTorch allocator to log per-step allocation/free events as Parquet traces.
- **`auto_instrument`** — Wrapper that attaches the monitor to a training loop: `model, optimizer = auto_instrument(model, optimizer)`
- **FastAPI REST API** — Exposes defrag status and telemetry endpoints
- **React Dashboard** — Vite+React frontend for VRAM visualization

---

## Status & Limitations

This is a **research prototype**, not production infrastructure. Key limitations:

- **Optimizer state is NOT compacted.** Only `model.parameters()` are repacked. Adam's `exp_avg`/`exp_avg_sq` remain in their original allocations. A complete solution would walk `optimizer.state_dict()` and include state tensors in the compaction buffer.
- **Gradients are not migrated.** `p.grad` tensors are separate allocations not included in the contiguous buffer.
- **DDP barrier from daemon thread is unsafe.** Calling `torch.distributed.barrier()` from a background thread can deadlock NCCL collectives. The DDP path should only be invoked from the main training loop.
- **Benchmarks are simulated.** The numbers in `benchmarks/` come from synthetic fragmentation curves, not real GPU measurements. Real validation requires running on actual hardware with `torch.cuda.memory_stats()`.
- **The FragPredictor has no training script.** The Transformer model exists but has no labeled dataset or training loop. The system falls back to the rule-based `OOMRiskModel` heuristic in practice.

To run a real measurement, use a Colab T4 and compare `torch.cuda.memory_stats()['allocated_bytes.all.peak']` with and without the defragmenter on the same training loop.

---

## Quick Start

### Install

```bash
git clone https://github.com/poojakira/RTX-OOM-Guard.git
cd RTX-OOM-Guard
pip install -e .
```

### Zero-Code-Change Integration

```python
from rtx_oom_guard import auto_instrument
model, optimizer = auto_instrument(model, optimizer)
# ... standard training loop, no other changes needed
```

### Manual Monitor

```python
from rtx_oom_guard import DefragMonitor
monitor = DefragMonitor(threshold=0.7)
monitor.start()
for batch in dataloader:
    monitor.record_alloc(tensor.numel() * tensor.element_size())
    output = model(batch)
    loss.backward()
    optimizer.step()
monitor.stop()
print(monitor.stats())
```

### Docker

```bash
docker build -t rtx-oom-guard .
docker run --gpus all rtx-oom-guard
```

### React Dashboard

```bash
cd dashboard
npm install
npm run dev  # http://localhost:5173
```

---

## Configuration

Edit `configs/config.yaml`:

```yaml
defrag:
  threshold: 0.7       # Fragmentation score to trigger compaction
  interval_ms: 50      # Monitor polling interval
  cooldown_steps: 10   # Steps between compaction runs
  use_triton: true     # Use Triton kernels if available
logging:
  results_dir: results
```

---

## Project Structure

```
.
├── src/rtx_oom_guard/
│   ├── defrag_engine/     # GPUMemoryDefragmenter, compactor, policy
│   ├── defrag/            # Custom Triton copy kernel
│   ├── scheduler/         # DefragMonitor, OOMRiskModel
│   ├── predictor/         # FragPredictor ML model
│   ├── profiler/          # AllocationCollector, AllocatorLogger
│   ├── trainer/           # auto_instrument, DefragCallback, DDPSyncManager
│   ├── llm_system/        # KV cache manager
│   └── api/main.py        # FastAPI REST API
├── dashboard/            # React + Vite frontend (13 panels)
├── benchmarks/           # OOM benchmarks, model fragmentation tests
├── data/traces/          # 100+ Parquet memory trace files
├── results/              # Benchmark results
├── tests/                # 50+ test files
├── configs/config.yaml
├── Dockerfile
└── run_benchmark.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Design Decisions

**Transformer predictor over simple heuristics** — `FragPredictor` in `src/rtx_oom_guard/predictor/` uses a Transformer architecture rather than a threshold on current fragmentation ratio. Fragmentation is path-dependent: the same 60% free memory can be perfectly contiguous or fatally scattered depending on the allocation/free sequence that produced it. A Transformer's self-attention over the recent allocation event stream captures these temporal patterns (e.g., repeated alloc-free-alloc cycles that create Swiss-cheese holes) which a single-point heuristic cannot.

**Thread-safety in telemetry persistence** — The telemetry writer originally spawned a new `threading.Thread` per write to avoid blocking the monitor's 50 ms polling loop. Under high-frequency monitoring this leaked hundreds of threads per second, exhausting OS limits. The fix was switching to synchronous atomic writes (`os.replace` on a temp file) inside the monitor thread itself — the I/O is fast enough at the granularity of one JSON blob per interval, and eliminates the thread lifecycle overhead entirely.

**Tiered policy: compact → evict → emergency** — A single fragmentation threshold would either trigger too early (wasting compaction cycles on benign fragmentation) or too late (OOM already inevitable). The tiered policy in `src/rtx_oom_guard/defrag_engine/` gives the system graduated responses: light compaction at 0.7, tensor eviction to CPU at 0.85, and emergency full-stop GC + compaction at 0.95. Each tier is cheaper than the next, so the common case (mild fragmentation) pays minimal overhead.

**Tensor registration requirement** — The system requires explicit tensor registration via `monitor.register_tensors()` (or implicitly through `auto_instrument`). Passive CUDA memory monitoring can detect fragmentation but cannot defragment without knowing which live tensors to relocate — PyTorch's caching allocator doesn't expose a handle→tensor mapping. Registration builds that mapping so the compactor can safely `.data`-swap tensors into contiguous blocks without breaking autograd references.

---

## Roadmap

- [ ] Automatic Triton kernel tuning per GPU model
- [ ] Integration with HuggingFace Trainer as a callback
- [ ] Support for FSDP (Fully Sharded Data Parallel)
- [ ] Live memory visualization via WebSocket
- [ ] PyPI package release

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

Built by [Pooja Kiran](https://github.com/poojakira) — M.S. student at Arizona State University.
