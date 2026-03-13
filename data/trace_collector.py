import torch
import pandas as pd
import os
import sys
import time
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.getcwd())

class ManualTraceCollector:
    def __init__(self, model_name):
        self.model_name = model_name
        self.events = []
        self.last_mem = 0

    def record_if_changed(self):
        curr_mem = torch.cuda.memory_allocated()
        if curr_mem != self.last_mem:
            self.events.append({
                "size": curr_mem - self.last_mem,
                "action": 1 if curr_mem > self.last_mem else 0,
                "timestamp": time.time() * 1000
            })
            self.last_mem = curr_mem

def collect_manual_traces(model_name, iterations=200):
    print(f"Collecting manual traces for {model_name}...")
    device = "cuda"
    if model_name == "gpt2":
        from benchmark.run_baseline import SimpleGPT2
        model = SimpleGPT2(n_layers=6).to(device)
        inputs = torch.randint(0, 50257, (8, 512), device=device)
    elif model_name == "resnet50":
        from torchvision.models import resnet50
        model = resnet50().to(device)
        inputs = torch.randn(16, 3, 224, 224, device=device)
    elif model_name == "bert":
        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True),
            num_layers=6
        ).to(device)
        inputs = torch.randn(8, 512, 768, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    collector = ManualTraceCollector(model_name)

    for i in range(iterations):
        optimizer.zero_grad()
        collector.record_if_changed()
        outputs = model(inputs)
        collector.record_if_changed()
        loss = outputs.sum() if isinstance(outputs, torch.Tensor) else outputs[0].sum()
        collector.record_if_changed()
        loss.backward()
        collector.record_if_changed()
        optimizer.step()
        collector.record_if_changed()
        if i % 50 == 0:
            print(f"  Iteration {i}")

    if not collector.events:
        print("Warning: No events captured.")
        return 0

    df = pd.DataFrame(collector.events)
    os.makedirs("data/traces", exist_ok=True)
    filename = f"data/traces/{model_name}_manual.parquet"
    df.to_parquet(filename)
    print(f"Saved {len(collector.events)} events to {filename}")
    return len(collector.events)

if __name__ == "__main__":
    total = 0
    for m in ["gpt2", "resnet50", "bert"]:
        try:
            total += collect_manual_traces(m)
        except Exception as e:
            print(f"Error collecting {m}: {e}")
    print(f"Total manual events: {total}")
