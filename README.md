# Predictive GPU Memory Defragmenter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch research prototype that **predicts GPU memory fragmentation and triggers proactive compaction before Out-of-Memory (OOM) crashes occur** during deep learning training.

Tested on: NVIDIA RTX 4060 8 GB · PyTorch 2.6.0.dev · CUDA 12.1 · GPT-2 (6-layer, 768-dim)

---

## The Problem

GPU memory fragmentation is a frequent cause of training instability. PyTorch's caching allocator reserves large blocks over time, but repeated alloc/free cycles produce small, scattered free regions that cannot satisfy the next large allocation — resulting in an OOM crash even when `nvidia-smi` shows free memory. The standard fix (`torch.cuda.empty_cache()`) is reactive: it runs *after* the failure. This project explores whether a predictive approach can act *before* the crash.

---

## What Was Built

### 1. Fragmentation Predictor (`gpudefrag/scheduler/predictor.py`)

A lightweight Transformer encoder (`FragPredictor`) that takes a sliding window of the last `seq_len=64` allocation events as input features `(batch, seq_len, 4)` and outputs a scalar fragmentation score in `[0, 1]` via sigmoid regression.

Architecture:
- Linear input projection → LayerNorm → GELU
- Learnable positional encoding `(1, seq_len, hidden_dim=128)`
- N=4 Transformer encoder layers with pre-norm (`norm_first=True`)
- Global average pooling across the time axis
- 3-layer regression head → sigmoid score
- Xavier uniform initialization throughout
- ~500K trainable parameters total

When the predicted score exceeds `risk_threshold=0.8`, the monitor thread triggers a compaction sweep.

### 2. Triton Defragmentation Kernels (`gpudefrag/defrag_engine/kernels.py`)

Two custom Triton kernels:

- `_compaction_copy_kernel` — copies tensor data from a potentially fragmented source into a fresh contiguous CUDA buffer using 1024-element blocks, maximizing memory bandwidth. This is the physical compaction step.
- `_fragmentation_scan_kernel` — a parallel scan over an allocator block-size array that computes local fragmentation scores on the GPU, avoiding CPU round-trip overhead.

Both kernels fall back gracefully to a `DummyTriton` stub when Triton is not installed, so the package runs on CPU-only environments for testing.

### 3. Defragmentation Engine (`gpudefrag/defrag_engine/defragmenter.py`)

`GPUMemoryDefragmenter` iterates over live model parameters, checks whether each tensor is already contiguous, and if not uses the Triton compaction copy (or a `torch.clone()` fallback) to repack it into a fresh contiguous VRAM block. Parameter data pointers are updated in-place so the autograd computation graph remains valid.

### 4. Zero-Code-Change Auto-Instrumentation (`gpudefrag/trainer/auto_instrument.py`)

Dynamically intercepts forward passes, backward passes, and optimizer steps using PyTorch hooks, abstracting the `TrainingHook` entirely away from the user's training code:

```python
from gpudefrag import auto_instrument
model, optimizer = auto_instrument(model, optimizer, risk_threshold=0.8)
# Existing training loop runs unchanged below
```

### 5. DDP Safety (`gpudefrag/trainer/ddp.py`)

`DDPSyncManager` wraps multi-GPU compaction events with `torch.distributed.barrier()` synchronization and `all_reduce(MAX)` risk checks to prevent NCCL hangs during compaction across ranks.

### 6. Monitoring Dashboard (`dashboard/`)

A React + FastAPI real-time dashboard with six inspection pages:

| Page | What it shows |
|---|---|
| Mission Control | OOM crashes prevented, cumulative VRAM recovered |
| VRAM Topology | Live hex-offset physical memory layout map |
| Shadow Forecast | Predicted fragmentation timeline with OOM threshold overlay |
| Scheduler Attention | Allocator decision heatmap |
| DDP Choreography | Multi-GPU barrier sync status and overhead |
| Triton Inspector | Kernel-level latency profiling and compaction traces |

---

## Benchmark Results

Setup: RTX 4060 (8 GB VRAM), PyTorch 2.6.0.dev, CUDA 12.1, GPT-2 (6-layer, 768-dim), 100 training iterations with synthetic fragmentation injected.

| Metric | Baseline | With gpudefrag | Change |
|---|---|---|---|
| OOM Errors | 0–3 per run | 0 | Eliminated |
| Training Restarts | 2–5 | 0 | Eliminated |
| Peak Memory (MB) | 7,840.4 | 6,920.4 | −11.7% |
| Avg Iteration Time | 1.94 s | 1.76 s | −9.3% |
| Proactive Compactions | — | 42 per session | Automatic |
| Triton Sweep Latency | — | 7.3–14.5 ms | Sub-iteration |
| Test Coverage | — | 100% (267 tests) | — |

Full numbers and raw JSON outputs are in [RESULTS.md](RESULTS.md). Reproduction steps are in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

**Honest caveats:**
- All benchmarks use *synthetic* fragmentation injection, not organic fragmentation from real production workloads.
- Tested on a single consumer GPU (RTX 4060). Multi-GPU DDP paths are implemented but not benchmarked end-to-end.
- The `FragPredictor` was trained on traces from this specific workload; out-of-distribution generalization to other architectures is not yet validated.
- This is a research prototype, not a drop-in production memory manager.

---

## Quick Start

```bash
pip install -e ".[models]"

# Profile a model
gpu-defragger profile --model gpt2

# Start the FastAPI telemetry server
gpu-defragger server --port 8000

# Launch the monitoring dashboard (React dev server at http://localhost:5173)
gpu-defragger dashboard

# Run the benchmark
python benchmarks/compare.py
```

---

## Repository Structure

```
gpudefrag/
├── profiler/
│   ├── collector.py          # torch.cuda.memory_snapshot ingestion
│   └── allocator_logger.py   # High-resolution allocation event logger
├── scheduler/
│   ├── predictor.py          # FragPredictor: Transformer fragmentation forecaster
│   ├── dataset.py            # Trace-to-tensor dataset pipeline
│   ├── risk_model.py         # Multi-signal OOM risk scorer
│   └── monitor.py            # Background DefragMonitor thread
├── defrag_engine/
│   ├── kernels.py            # Triton kernels: compaction copy + fragmentation scan
│   ├── defragmenter.py       # GPUMemoryDefragmenter: active VRAM repacker
│   └── policy.py             # MitigationPolicy decision engine
├── trainer/
│   ├── auto_instrument.py    # Zero-code-change PyTorch hook orchestrator
│   ├── training_hook.py      # Low-level forward/backward/optimizer interceptors
│   └── ddp.py                # DDPSyncManager: multi-GPU barrier choreography
├── optimization/             # int8 quantization utilities
├── llm_system/               # PagedKV cache integration
├── api.py                    # FastAPI telemetry REST surface
├── cli.py                    # Rich-powered CLI
└── dashboard.py              # Dashboard bridge/launcher

benchmarks/
├── compare.py                # Baseline vs defrag report generator
└── run_local_benchmark.py    # Per-rank local simulation engine

tests/                        # 267 unit tests, 100% statement coverage
results/                      # Benchmark JSON and CSV outputs
data/traces/                  # Allocation event trace data
checkpoints/                  # FragPredictor model checkpoints
dashboard/                    # React frontend
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Core ML / runtime | PyTorch |
| Prediction model | Transformer encoder (PyTorch `nn.TransformerEncoder`) |
| GPU kernels | Triton (`triton_compaction_copy`, `_fragmentation_scan_kernel`) |
| API | FastAPI |
| Dashboard | React + Vite |
| CLI | Rich |
| Testing | pytest (267 tests, 100% statement coverage) |

---

## Engineers

**Pooja Kiran** — ML systems: `FragPredictor` Transformer architecture and training, allocator telemetry pipeline, Triton kernel authorship, `GPUMemoryDefragmenter` engine, `DDPSyncManager` integration, `auto_instrument` hook orchestration.

**Rhutvik Pachghare** — Observability: AeroGrid dashboard architecture and React frontend, hardware visualization topology, CI/CD pipeline and repository hardening.

---

## Reproduce Results

```bash
pip install -e ".[models]"
python benchmarks/compare.py
# Outputs: results/comparison.csv, results/comparison.json
```

---

## License

MIT — see [LICENSE](LICENSE)
