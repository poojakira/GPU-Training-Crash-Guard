# gpudefrag — Predictive GPU Memory Defragmenter

**Production-grade infrastructure for modeling GPU memory fragmentation, predicting OOM failures, and implementing proactive mitigation strategies for PyTorch training workflows.**

gpudefrag is a modular ML infrastructure tool that profiles GPU memory allocator behavior, predicts fragmentation-induced OOM risk, and applies configurable mitigation policies to prevent training failures. The tool includes real-world training trace collection, a Transformer-based fragmentation predictor, and a React dashboard for live monitoring.

**Core features:** allocator-state profiling · OOM-risk prediction · PyTorch training hooks · automated mitigation policy · benchmark replay · real-time dashboard

---

## 1. Overview

This project provides an end-to-end pipeline for GPU memory observability and proactive defragmentation during DL training. It captures real memory telemetry from actual model training (GPT-2, ResNet-50, BERT), trains a Transformer predictor on the observed patterns, and provides hooks to automatically intervene before OOM failures.

### Key capabilities

| Component | Description |
|---|---|
| **Allocator Logger** (`src/monitor/`) | Per-step memory telemetry: `allocated`, `reserved`, `fragmentation_ratio`, `step_time`, `batch_size` |
| **OOM Risk Predictor** (`src/predictor/`) | Rule-based + logistic regression risk scorer with configurable thresholds |
| **Training Hooks** (`src/hooks/`) | PyTorch-native hooks for forward/backward/optimizer phases with memory injection |
| **Mitigation Policy** (`src/mitigation/`) | 3-tier policy engine: SAFE → WARN → ACT (empty_cache + batch-size downshift) |
| **Transformer Predictor** (`gpudefrag/scheduler/`) | Sequence model trained on allocation traces to forecast fragmentation |
| **Workload Simulator** (`scripts/`) | High-fidelity GPU memory simulator with block-level allocator model |
| **Real Trace Collector** (`scripts/`) | Captures genuine memory data from actual PyTorch model training |
| **Dashboard** (`dashboard/`) | React/Vite dashboard with fragmentation charts, memory maps, and KPI grids |
| **FastAPI Surface** (`gpudefrag/api.py`) | REST endpoints: `/health`, `/memory`, `/risk`, `/risk/history` |

---

## 2. Evaluation Results

Benchmark results from 5 independent runs on RTX 4060 hardware:

| Metric | Vanilla PyTorch | gpudefrag | Improvement |
|---|---:|---:|---:|
| **OOM Errors** | 4.0 ± 0.3 | 0.0 ± 0.0 | **100.0%** |
| **Iteration Time (s)** | 2.53 ± 0.10 | 1.83 ± 0.05 | **27.6% faster** |
| **Peak Memory (MB)** | 7,192 ± 40 | 6,620 ± 22 | **7.9% reclaimed** |
| **Throughput (iter/s)** | 0.39 | 0.54 | **+38%** |
| **Avg Fragmentation** | — | 0.150 | — |

> **Note:** Results measured on RTX 4060 development environment. CPU-simulated benchmarks are included for CI validation.

### Saved results

- `results/run_1.json` through `results/run_5.json` — per-step memory logs with fragmentation ratios
- `results/summary.csv` — aggregated metrics across all runs
- `results/plots/fragmentation_vs_time.png` — visualization of fragmentation over training steps

---

## 3. Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop (PyTorch)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Forward  │→ │ Backward │→ │ Optimizer│→ │ Zero Grad  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────────┘  │
│       │              │              │                        │
│       ▼              ▼              ▼                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            TrainingHook (src/hooks/)                  │   │
│  │  on_forward_begin/end · on_backward · on_optimizer   │   │
│  └──────────────┬───────────────────────────────────────┘   │
│                 │                                            │
│       ┌─────────┴──────────┐                                │
│       ▼                    ▼                                │
│  ┌──────────┐     ┌──────────────┐    ┌─────────────────┐  │
│  │Allocator │     │ OOM Risk     │    │ Mitigation      │  │
│  │ Logger   │     │ Model        │    │ Policy          │  │
│  │(monitor/)│     │(predictor/)  │    │(mitigation/)    │  │
│  └──────────┘     └──────────────┘    └─────────────────┘  │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  gpudefrag/ (core package)                                   │
│  ├── profiler/    — CUDA allocator snapshot capture           │
│  ├── scheduler/   — FragPredictor (Transformer) + dataset    │
│  ├── engine/      — Defrag Triton kernels                    │
│  ├── trainer/     — Model training pipeline                  │
│  └── api.py       — FastAPI endpoints                        │
├──────────────────────────────────────────────────────────────┤
│  dashboard/       — React/Vite visualization                 │
│  benchmarks/      — Local RTX benchmark runner               │
│  scripts/         — Workload simulator + real trace collector │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Quick Start

### Install

```bash
git clone https://github.com/poojakira/Predictive-GPU-Memory-Defragmenter.git
cd Predictive-GPU-Memory-Defragmenter
pip install -e .
```

### Run benchmark

```bash
python benchmarks/run_local_benchmark.py --runs 5 --steps 100
```

### Run example

```bash
python examples/train_toy_model.py
```

### Run API server

```bash
gpudefrag-api
# or: uvicorn gpudefrag.api:app --host 0.0.0.0 --port 8000
```

### Run dashboard

```bash
cd dashboard && npm install && npm run dev
```

### Collect real training traces

```bash
python scripts/collect_real_traces.py --steps 500 --models gpt2 resnet50 bert --vary-batch
```

### Generate synthetic workload traces

```bash
python scripts/generate_senior_dataset.py --count 150 --steps 10000
```

---

## 5. Testing

```bash
# Core unit tests (25 tests)
pytest tests/test_logger.py tests/test_risk_model.py tests/test_hook.py -v

# Workload simulator tests (21 tests)
pytest tests/test_simulator.py -v

# Full suite
pytest tests/ -v
```

All tests run on **CPU-only** environments — no GPU required for CI.

---

## 6. Repository Structure

```text
.github/
  workflows/ci.yml       Multi-Python CI pipeline (3.10–3.12)
  CODEOWNERS             Code ownership assignments
benchmark/               Legacy benchmark assets
benchmarks/
  run_local_benchmark.py RTX benchmark runner (CPU fallback)
dashboard/               React/Vite monitoring dashboard
data/traces/             Parquet traces (synthetic + real)
examples/
  train_toy_model.py     End-to-end demo
gpudefrag/               Core ML infrastructure package
  api.py                 FastAPI endpoints
  profiler/              Allocator snapshot capture
  scheduler/             FragPredictor + dataset pipeline
  engine/                Defrag kernel experiments
  trainer/               Training pipeline
scripts/
  workload_simulator.py  High-fidelity GPU memory simulator
  collect_real_traces.py Real PyTorch training trace collector
  generate_senior_dataset.py  Batch trace generation
  train_senior_predictor.py   Transformer training on traces
src/
  monitor/               AllocatorLogger
  predictor/             OOMRiskModel
  hooks/                 TrainingHook
  mitigation/            MitigationPolicy
tests/                   Unit + integration tests
results/                 Benchmark outputs (JSON, CSV, plots)
```

---

## 7. Code Ownership

| Area | Owner |
|---|---|
| ML Engineering (`src/`, `gpudefrag/`, `scripts/`, `benchmarks/`, `tests/`) | **Pooja Kiran** (`@poojakira`) |
| Robotics Engineering (`dashboard/`, `.github/`, `benchmark/`) | **Rhutvik Pachghare** (`@rhutvik-pachghare`) |

See `.github/CODEOWNERS` for automatic review assignments.

---

## 8. References

- **PyTorch CUDA Allocator:** https://pytorch.org/docs/stable/notes/cuda.html
- **Triton:** https://triton-lang.org/
- **NVIDIA Memory Management:** CUDA best practices documentation

---

**Version:** v3.0.0  
**License:** MIT  
**Hardware tested:** RTX 4060  
**CI validated:** Python 3.10 – 3.12 (CPU-only)
