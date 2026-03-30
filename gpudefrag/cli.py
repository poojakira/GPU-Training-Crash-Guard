"""
gpudefrag.cli — Enterprise Command-Line Interface.

NVIDIA-grade CLI entry point for the Predictive GPU Memory Defragmenter.
Provides a unified interface for profiling, simulating, and auto-instrumenting training runs.
"""

import argparse
import sys
import os
import time

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich import print as rprint
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_banner():
    """Prints an NVIDIA-style enterprise banner."""
    if HAS_RICH:
        banner = Text("▶ gpudefrag: NVIDIA-Grade GPU Memory Infrastructure", style="bold #76b900")
        console.print(Panel(banner, border_style="#76b900"))
    else:
        print("="*60)
        print("▶ gpudefrag: NVIDIA-Grade GPU Memory Infrastructure")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="gpudefrag: NVIDIA-Grade Predictive Memory Defragmenter Engine"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")
    
    # 1. Profile command
    profile_p = subparsers.add_parser("profile", help="Collect raw VRAM telemetry from reference models")
    profile_p.add_argument("--model", choices=["gpt2", "resnet50", "bert", "all"], default="all")
    profile_p.add_argument("--iterations", type=int, default=200)
    
    # 2. Simulate command
    sim_p = subparsers.add_parser("simulate", help="Run the benchmark simulation locally without GPUs")
    sim_p.add_argument("--baseline", action="store_true", help="Run baseline without mitigation")
    
    # 3. Server command
    serve_p = subparsers.add_parser("server", help="Launch the local live Telemetry API server")
    serve_p.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    print_banner()

    if args.command == "profile":
        if HAS_RICH:
            console.print(f"[bold cyan]▶ Starting telemetry collection for {args.model}...[/]")
        else:
            print(f"▶ Starting telemetry collection for {args.model}...")
        
        # We can trigger the existing python scripts locally 
        # (This avoids circular imports in this CLI wrapper)
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "collect_real_traces.py")
        os.system(f"{sys.executable} {script_path}")
        
    elif args.command == "simulate":
        if HAS_RICH:
            console.print("[bold #76b900]▶ Launching Hardware Workload Simulation...[/]")
        else:
            print("▶ Launching Hardware Workload Simulation...")
            
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks", "run_local_benchmark.py")
        os.system(f"{sys.executable} {script_path}")

    elif args.command == "server":
        if HAS_RICH:
            console.print(f"[bold cyan]▶ Starting gpudefrag REST API on port {args.port}...[/]")
        else:
            print(f"▶ Starting gpudefrag REST API on port {args.port}...")
            
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gpudefrag", "api.py")
        os.system(f"{sys.executable} -m uvicorn gpudefrag.api:app --host 0.0.0.0 --port {args.port} --reload")

if __name__ == "__main__":
    main()
