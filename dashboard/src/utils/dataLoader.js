// Process processed chart data from snapshots
const processChartSnapshots = (snapshots) => {
  return snapshots.map(s => ({
    iteration: s.iteration,
    frag: (s.frag * 100).toFixed(1),
    allocated: s.allocated_mb.toFixed(0)
  }));
};

// Fetch live telemetry (Real-time)
export const fetchLiveTelemetry = async () => {
  try {
    const response = await fetch('/live/live_telemetry.json?t=' + Date.now());
    if (!response.ok) throw new Error('Live telemetry not found');
    const data = await response.json();
    
    return {
      currentAllocated: data.current_allocated_mb.toFixed(0),
      currentReserved: data.current_reserved_mb.toFixed(0),
      currentFrag: (data.current_frag * 100).toFixed(1),
      totalCompactions: data.total_compactions,
      totalFreed: data.total_freed_mb.toFixed(0),
      history: data.compaction_history.map(c => ({
        id: c.compaction_id,
        freed: c.freed_mb,
        fragReduction: (c.frag_reduction * 100).toFixed(1),
        elapsedMs: c.elapsed_ms.toFixed(0),
        timestamp: c.timestamp
      })),
      avgTime: data.avg_latency_ms.toFixed(3) // Monitoring latency
    };
  } catch (e) {
    console.warn("Polling live data failed, system might be idle:", e.message);
    return null;
  }
};

// Fetch baseline/static stats
export const fetchBenchmarkStats = async () => {
    try {
        const [baseRes, defragRes] = await Promise.all([
            fetch('/live/baseline.json'),
            fetch('/live/defrag.json')
        ]);
        
        const base = await baseRes.json();
        const defrag = await defragRes.json();
        
        return {
            baseline: {
                avgTime: base.avg_iteration_time.toFixed(2),
                peakMem: base.peak_memory_mb.toFixed(0),
                chart: base.memory_snapshots.map(s => ({ iteration: s.iteration, frag: (s.frag * 100).toFixed(1) }))
            },
            defrag: {
                avgTime: defrag.avg_iteration_time.toFixed(2),
                peakMem: defrag.peak_memory_mb.toFixed(0),
                chart: defrag.memory_snapshots.map(s => ({ iteration: s.iteration, frag: (s.frag * 100).toFixed(1) }))
            }
        };
    } catch (e) {
        console.error("Static data load failed:", e);
        return null;
    }
};
