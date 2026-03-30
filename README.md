# Predictive GPU Memory Defragmenter 

An enterprise-grade, "Zero-Code-Change" PyTorch ML Infrastructure tool designed to actively predict and mitigate GPU memory fragmentation before Out-of-Memory (OOM) exceptions occur. 

## 🏆 Project Engineering Leads

* **Pooja Kiran** — ML Engineering Lead  
  *Architect of the Predictive Risk Models, Core Telemetry Pipelines, PyTorch Allocator Hooking, native Triton-powered active tensor defragmentation kernels, and Distributed Data Parallel (DDP) integration logic.*
* **Rhutvik Pachghare** — Robotics & UI Engineering Lead  
  *Architect of the NVIDIA-Nsight Cinematic Dashboard, hardware visualization topology logic, and CI/CD operations.*

---

## ⚡ Key Enterprise Features

1. **Zero-Code-Change Auto-Instrumentation**
   Forget plastering `hook.on_forward_begin()` across your codebase. Wrap your model once:
   ```python
   from gpudefrag.trainer import auto_instrument
   model, optimizer = auto_instrument(model, optimizer, risk_threshold=0.8)
   ```
2. **True Triton-Powered GPU Defragmentation**
   Unlike simple `empty_cache()` scripts, this engine acts as a **true physical defragmenter**. It uses extreme-bandwidth custom Triton kernels (`triton_compaction_copy`) to seamlessly repack live model parameters into dense VRAM blocks without severing autograd backward graphs.
3. **NVIDIA Nsight-Themed Dashboard**
   A cinematic, high-density React front-end ("Mission Control") visualizing live fragmentation streams, Hex-Offset VRAM Address Topology, and hardware metrics.
4. **Distributed Data Parallel (DDP) Safe**
   Includes native `torch.distributed.barrier()` safety nets to prevent NCCL broadcast hangs when multi-GPU synchronization requires live memory compaction.

---

## 🚀 Quick Start

### 1. Unified Command Line Interface (CLI)

The package provides a seamless Rich-powered CLI:

```powershell
# Profile real models (gpt2, resnet50)
python -m gpudefrag.cli profile --model gpt2

# Launch Live REST API Server
python -m gpudefrag.cli server --port 8000

# Simulate Workloads
python -m gpudefrag.cli simulate
```

### 2. Launching the Enterprise Dashboard

The React Mission Control dashboard runs independently:

```powershell
cd dashboard
npm install
npm run dev
```
Navigate to `http://localhost:5173/` to view the active UI.

---

## 🏗️ Architecture

- **`gpudefrag/trainer/auto_instrument.py`**: Intersects PyTorch hook dispatches invisibly.
- **`gpudefrag/defrag_engine/defragmenter.py`**: The true VRAM memory-repacking logic engine.
- **`src/predictor/`**: Predictive Machine Learning models (LSTMs/RandomForests) forecasting OOM boundaries based on trace history.
- **`dashboard/`**: React/Vite front-end plotting real-time data from `dashboard/public/live/telemetry.json`.
