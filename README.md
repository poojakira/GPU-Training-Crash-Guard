# Predictive GPU Memory Defragmenter

A Transformer-based proactive GPU memory defragmenter for PyTorch/NVIDIA workflows.

## Overview

This project implements a system that:
1.  **Collects** allocation traces from PyTorch workloads.
2.  **Predicts** future memory fragmentation using a Transformer model.
3.  **Proactively Defragments** GPU memory before Out-of-Memory (OOM) errors occur.

## Setup

1.  **Environment**: Python 3.10+
2.  **Installation**:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Verify GPU**:
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```

## Project Structure

- `benchmark/`: Evaluation scripts and baselines.
- `data/`: Allocation traces and dataset processing.
- `model/`: Transformer predictor architecture.
- `defrag/`: Monitoring and compaction logic.
- `results/`: Benchmark data and comparison reports.
- `checkpoints/`: Trained model weights.

## Usage

(Details coming soon)
