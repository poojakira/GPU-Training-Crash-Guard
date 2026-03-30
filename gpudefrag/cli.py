"""
gpudefrag.cli — Command-line interface entry points.
"""

import argparse
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect_cmd():
    """CLI entry point for trace collection."""
    parser = argparse.ArgumentParser(description="Collect CUDA allocation traces")
    parser.add_argument("--model", choices=["gpt2", "resnet50", "bert", "all"], default="all")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output-dir", default="data/traces")
    args = parser.parse_args()

    from gpudefrag.profiler.collector import collect_from_model
    from gpudefrag.utils import DefragConfig

    config = DefragConfig(trace_dir=args.output_dir)
    models = ["gpt2", "resnet50", "bert"] if args.model == "all" else [args.model]
    total = 0

    for m in models:
        try:
            total += collect_from_model(m, iterations=args.iterations, config=config)
        except Exception as e:
            print(f"Error collecting {m}: {e}")

    print(f"\nTotal events collected: {total}")


def train_cmd():
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(description="Train the fragmentation predictor")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--trace-dir", default="data/traces")
    args = parser.parse_args()

    from gpudefrag.trainer.trainer import train
    from gpudefrag.utils import DefragConfig

    config = DefragConfig(
        train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        trace_dir=args.trace_dir,
    )
    train(config)


def benchmark_cmd():
    """CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="Run benchmark comparison")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--compare", action="store_true", help="Run both baseline and defrag")
    args = parser.parse_args()

    if args.compare:
        from benchmark.compare import run_comparison
        run_comparison(args.iterations)
    else:
        from benchmark.run_baseline import run_benchmark
        run_benchmark(args.iterations)


def main():
    """Main entry point for the productized gpu-defragger CLI."""
    parser = argparse.ArgumentParser(description="gpu-defragger: ML Infrastructure Memory Optimization")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run defragmenter using a YAML configuration")
    run_parser.add_argument("config", help="Path to config.yaml")
    
    args = parser.parse_args()
    
    if args.command == "run":
        import yaml
        from gpudefrag.utils import DefragConfig
        from gpudefrag.scheduler.monitor import DefragMonitor
        
        config_path = args.config
        if not os.path.exists(config_path):
            print(f"Error: Config file not found at {config_path}")
            sys.exit(1)
            
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f)
            
        # Merge YAML into DefragConfig
        cfg = DefragConfig()
        for k, v in yaml_data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
                
        print(f"Loaded configuration from {config_path}")
        print(f"Initializing gpu-defragger Monitor...")
        monitor = DefragMonitor(config=cfg)
        monitor.start()
        print("gpu-defragger is running in the background. Press Ctrl+C to stop.")
        
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping gpu-defragger...")
            monitor.stop()
            print("Stopped.")

if __name__ == "__main__":
    main()
