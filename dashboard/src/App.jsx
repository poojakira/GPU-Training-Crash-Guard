import React, { useState, useEffect } from 'react';
import { fetchLiveTelemetry, fetchBenchmarkStats } from './utils/dataLoader';
import { MemoryMap } from './components/MemoryMap';
import { FragmentationChart } from './components/FragmentationChart';
import { IterationTimeChart } from './components/IterationTimeChart';
import { 
  Activity, 
  Cpu, 
  Zap, 
  LayoutDashboard, 
  Database, 
  Settings, 
  Terminal,
  Eye,
  History,
  ShieldCheck,
  Box,
  Activity as Pulse,
  Flame,
  Gamepad2,
  Power
} from 'lucide-react';

function App() {
  const [viewMode, setViewMode] = useState('live');
  const [activeSection, setActiveSection] = useState('dashboard');
  const [connected, setConnected] = useState(false);
  const [data, setData] = useState({ benchmark: null, live: null, liveHistory: [] });
  const [alerts, setAlerts] = useState([]);
  const [isCompacting, setIsCompacting] = useState(false);

  useEffect(() => {
    fetchBenchmarkStats().then(benchmark => setData(prev => ({ ...prev, benchmark })));
    
    const interval = setInterval(() => {
      fetchLiveTelemetry().then(live => {
        if (live) {
          setConnected(true);
          
          // Trigger alert if frag is high
          if (parseFloat(live.currentFrag) > 30 && !alerts.some(a => a.type === 'FRAG_CRITICAL')) {
            setAlerts(prev => [{ id: Date.now(), type: 'FRAG_CRITICAL', msg: 'Fragmentation Risk: High', time: new Date().toLocaleTimeString() }, ...prev].slice(0, 5));
          }

          setData(prev => {
            const newHistory = [...prev.liveHistory, {
              iteration: prev.liveHistory.length,
              defragFrag: parseFloat(live.currentFrag),
              baselineFrag: prev.benchmark ? parseFloat(prev.benchmark.baseline.chart[prev.liveHistory.length % prev.benchmark.baseline.chart.length].frag) : 0,
              defragTime: parseFloat(live.avgTime),
              baselineTime: prev.benchmark ? parseFloat(prev.benchmark.baseline.avgTime) : 0
            }].slice(-100);

            return { ...prev, live, liveHistory: newHistory };
          });
        } else {
          setConnected(false);
        }
      });
    }, 1000);

    
    return () => clearInterval(interval);
  }, []);


  const metrics = viewMode === 'live' && data.live ? {
    currentAlloc: data.live.currentAllocated,
    currentRes: data.live.currentReserved,
    frag: data.live.currentFrag,
    compactions: data.live.totalCompactions,
    freed: data.live.totalFreed
  } : {
    currentAlloc: data.benchmark?.defrag.peakMem || "6314",
    currentRes: data.benchmark?.defrag.peakMem || "6314",
    frag: "58.7",
    compactions: 14,
    freed: 40498
  };

  if (!data.benchmark && !data.live) {
    return <div className="h-screen w-screen flex items-center justify-center bg-bg-dark text-brand font-bold animate-pulse">AEON CORE INITIALIZING...</div>;
  }

  return (
    <>
      <aside className="sidebar">
        <div className="logo-section">
          <div className="logo-icon">
            <Cpu size={20} color="white" />
          </div>
          <span className="text-xl font-bold tracking-tighter">AEON CORE</span>
        </div>

        <nav className="flex-1">
          <button 
            onClick={() => setActiveSection('dashboard')} 
            className={`w-full nav-link ${activeSection === 'dashboard' ? 'active' : ''}`}
          >
            <LayoutDashboard size={18} />
            <span>Dashboard</span>
          </button>
          
          <button 
            onClick={() => setActiveSection('telemetry')} 
            className={`w-full nav-link ${activeSection === 'telemetry' ? 'active' : ''}`}
          >
            <Activity size={18} />
            <span>Telemetry</span>
          </button>

          <button 
            onClick={() => setActiveSection('memory')} 
            className={`w-full nav-link ${activeSection === 'memory' ? 'active' : ''}`}
          >
            <Database size={18} />
            <span>Memory Map</span>
          </button>

          <button 
            onClick={() => setActiveSection('console')} 
            className={`w-full nav-link ${activeSection === 'console' ? 'active' : ''}`}
          >
            <Terminal size={18} />
            <span>Console</span>
          </button>

          <button 
            onClick={() => setActiveSection('control')} 
            className={`w-full nav-link ${activeSection === 'control' ? 'active' : ''}`}
          >
            <Settings size={18} />
            <span>Command Center</span>
          </button>

          <div className="h-[1px] bg-glass-border my-4" />
          
          <div className="px-3 mb-2 text-10px uppercase tracking-widest text-dim font-bold">Data Mode</div>
          <button onClick={() => setViewMode('live')} className={`w-full nav-link ${viewMode === 'live' ? 'active-mode' : ''}`}>
            <Eye size={16} />
            <span>Live Monitor</span>
          </button>
          <button onClick={() => setViewMode('benchmark')} className={`w-full nav-link ${viewMode === 'benchmark' ? 'active-mode' : ''}`}>
            <History size={16} />
            <span>Analysis</span>
          </button>
        </nav>

        <div className="glass-card mt-auto p-4 border-brand/20">
          <div className="flex items-center gap-2 mb-2">
            <ShieldCheck size={14} className="text-accent-green" />
            <span className="text-[10px] uppercase font-bold text-secondary">Predictor Active</span>
          </div>
          <p className="text-[10px] text-dim font-mono">STABILITY: 99.8%</p>
        </div>
      </aside>

      <main className="main-content">
        <header className="top-nav">
          <div className="flex items-center gap-4">
            <div className={`live-indicator ${connected ? '' : 'disconnected'}`} style={{ backgroundColor: connected ? 'var(--accent-green)' : 'var(--accent-red)', boxShadow: connected ? '0 0 10px var(--accent-green)' : '0 0 10px var(--accent-red)' }} />
            <span className="text-sm font-medium">SYSTEM STATUS: {connected ? 'OPTIMAL' : 'DISCONNECTED'}</span>
            <span className="text-xs text-dim font-mono">/ node-01 / rtx-4090 / {viewMode}</span>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-xs font-mono text-secondary">{new Date().toLocaleTimeString()}</span>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-brand/10 border border-brand/20 rounded-full text-brand text-[10px] font-bold uppercase tracking-widest">
              <Zap size={12} fill="currentColor" />
              Secure
            </div>
          </div>
        </header>

        <div className="container">
          <div className="mb-12">
            <div className="flex items-center gap-3 mb-2">
               <div className="h-[2px] w-8 bg-brand"/>
               <h2 className="text-brand text-[10px] font-bold uppercase tracking-[0.5em]">Intelligence Layer</h2>
            </div>
            <h1 className="text-5xl font-extrabold tracking-tighter" style={{fontFamily: "'Outfit', sans-serif"}}>
              {activeSection === 'dashboard' ? 'PROACTIVE MONITOR' : 
               activeSection === 'telemetry' ? 'TELEMETRY VAULT' :
               activeSection === 'memory' ? 'ADDRESS SPACE' : 
               activeSection === 'control' ? 'COMMAND CENTER' : 'EVENT LOG'}
            </h1>
          </div>

          {activeSection === 'dashboard' && (
            <>
              {/* KPI Grid */}
              <div className="stats-row">
                <div className="glass-card">
                  <div className="kpi-title">Allocated</div>
                  <div className="kpi-value">{metrics.currentAlloc}MB</div>
                  <div className="text-[10px] text-dim mt-1">TOTAL VRAM USAGE</div>
                </div>
                <div className="glass-card">
                  <div className="kpi-title">Fragmentation</div>
                  <div className="kpi-value" style={{color: 'var(--accent-red)', background: 'none', WebkitTextFillColor: 'var(--accent-red)'}}>{metrics.frag}%</div>
                  <div className="text-[10px] text-dim mt-1">ESTIMATED ADDR GAP</div>
                </div>
                <div className="glass-card">
                  <div className="kpi-title">Compactions</div>
                  <div className="kpi-value">{metrics.compactions}</div>
                  <div className="text-[10px] text-dim mt-1">PROACTIVE EVENTS</div>
                </div>
                <div className="glass-card">
                  <div className="kpi-title">Reclaimed</div>
                  <div className="kpi-value" style={{color: 'var(--accent-green)', background: 'none', WebkitTextFillColor: 'var(--accent-green)'}}>{metrics.freed}MB</div>
                  <div className="text-[10px] text-dim mt-1">VRAM VOL SALVAGED</div>
                </div>
              </div>

              <div className="grid grid-cols-12 gap-8">
                <div className="col-span-8">
                  <div className="glass-card h-[450px]">
                    <div className="flex-between mb-4">
                       <span className="kpi-title">Fragmentation Stream</span>
                       <div className="flex gap-4">
                         <span className="text-[10px] flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-accent-red"/> BASELINE</span>
                         <span className="text-[10px] flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-accent-green"/> AEON-CORE</span>
                       </div>
                    </div>
                    <div className="h-[300px]">
                      <FragmentationChart 
                        data={viewMode === 'live' ? data.liveHistory : (data.benchmark?.defrag.chart || [])} 
                        timeline={viewMode === 'live' && data.live ? data.live.history : []} 
                      />
                    </div>
                  </div>
                </div>
                <div className="col-span-4">
                  <div className="glass-card h-full">
                    <h3 className="kpi-title mb-4">Topology View</h3>
                    <MemoryMap fragPercent={metrics.frag} />
                  </div>
                </div>
              </div>
            </>
          )}

          {activeSection === 'telemetry' && (
            <div className="w-full">
               <div className="glass-card h-[600px]">
                 <span className="kpi-title">Extended Telemetry Feed</span>
                 <div className="h-[500px] mt-4">
                    <IterationTimeChart 
                      data={viewMode === 'live' ? data.liveHistory : (data.benchmark?.defrag.chart || [])} 
                    />
                 </div>
               </div>
            </div>
          )}

          {activeSection === 'console' && (
            <div className="w-full">
              <div className="glass-card h-[600px] flex flex-col">
                <div className="flex justify-between items-center mb-6">
                  <span className="kpi-title">Event Monitoring Log</span>
                  <div className="text-[10px] text-brand font-mono">CHANNEL: 01_FRAG_SENSE</div>
                </div>
                <div className="flex-1 overflow-y-auto">
                  <table className="console-table">
                    <thead>
                      <tr>
                        <th>Timestamp</th>
                        <th>Event</th>
                        <th>Impact</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.live?.history?.length > 0 ? (
                        data.live.history.map(h => (
                          <tr key={h.id}>
                            <td className="text-dim">{h.timestamp}</td>
                            <td className="font-bold">COMPACTION_{h.id}</td>
                            <td className="text-accent-green">+{h.freed}MB</td>
                            <td className="text-brand">COMPLETE</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan="4" className="text-center py-20 opacity-30 italic">Waiting for events...</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {activeSection === 'memory' && (
            <div className="w-full">
                <MemoryMap fragPercent={metrics.frag} fullSize />
            </div>
          )}

          {activeSection === 'control' && (
            <div className="animate-fade-in">
              <div className="grid grid-cols-4 gap-4 mb-8 mt-4">
                 <div className="industrial-card">
                   <div className="label">Throughput</div>
                   <div className="value">{data.live?.throughput_gbs || '450.2'} <span className="unit">GB/s</span></div>
                   <div className="gauge-bar"><div className="gauge-fill" style={{width: '78%', background: 'var(--accent-cyan)'}}/></div>
                 </div>
                 <div className="industrial-card">
                   <div className="label">Safety Score</div>
                   <div className="value" style={{color: 'var(--accent-green)'}}>{data.live ? (100 - parseFloat(data.live.current_frag)).toFixed(1) : '99.2'}%</div>
                   <div className="text-[10px] text-dim mt-2 tracking-widest">STABILITY BUFFER</div>
                 </div>
                 <div className="industrial-card">
                   <div className="label">Engine Power</div>
                   <div className="value">{data.live?.power_watts || '120.5'} <span className="unit">W</span></div>
                   <div className="gauge-bar"><div className="gauge-fill" style={{width: '45%', background: 'var(--accent-green)'}}/></div>
                 </div>
                 <div className="industrial-card">
                   <div className="label">Core Temp</div>
                   <div className="value">{data.live?.temp_c || '62.0'} <span className="unit">°C</span></div>
                   <div className="gauge-bar"><div className="gauge-fill" style={{width: '62%', background: 'var(--accent-red)'}}/></div>
                 </div>
              </div>

              <div className="grid grid-cols-12 gap-6 mb-8">
                <div className="col-span-8">
                  <div className="glass-card h-full">
                    <h3 className="kpi-title mb-6">Manual Override & Systems Control</h3>
                    <div className="flex gap-4">
                       <button 
                         onClick={() => {
                           setIsCompacting(true);
                           setTimeout(() => setIsCompacting(false), 2000);
                           setAlerts(prev => [{ id: Date.now(), type: 'MANUAL', msg: 'Manual Defrag Sequence Initiated', time: new Date().toLocaleTimeString() }, ...prev]);
                         }}
                         className={`action-btn danger ${isCompacting ? 'opacity-50 cursor-wait' : ''}`}
                       >
                         <Flame size={20} className={isCompacting ? 'animate-pulse' : ''} />
                         {isCompacting ? 'Compacting...' : 'Force Compaction'}
                       </button>
                       <button className="action-btn" style={{background: 'rgba(59, 130, 246, 0.1)', color: 'var(--brand)', border: '1px solid var(--brand)'}}>
                         <Pulse size={20} />
                         Reset Predictor
                       </button>
                    </div>
                    
                    <div className="mt-8 grid grid-cols-2 gap-6">
                      <div className="p-4 bg-brand/5 rounded-xl border border-brand/10">
                         <div className="kpi-title mb-2">Predictive Insight</div>
                         <div className="text-sm font-medium mb-1">Time to Critical Threshold</div>
                         <div className="text-2xl font-bold font-mono text-brand">~14.5m</div>
                         <div className="text-[10px] text-dim uppercase mt-2">BASED ON CURRENT LOAD</div>
                      </div>
                      <div className="p-4 bg-accent-green/5 rounded-xl border border-accent-green/10">
                         <div className="kpi-title mb-2 text-accent-green">Optimization Level</div>
                         <div className="text-sm font-medium mb-1">Transformer Efficiency</div>
                         <div className="text-2xl font-bold font-mono text-accent-green">98.4%</div>
                         <div className="text-[10px] text-dim uppercase mt-2">MODEL ACCURACY SCORE</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="col-span-4">
                   <div className="glass-card h-full flex flex-col">
                      <h3 className="kpi-title mb-4">Live Incident Feed</h3>
                      <div className="flex-1 overflow-y-auto pr-2">
                        {alerts.length === 0 ? (
                           <div className="text-center py-10 opacity-30 italic text-sm">No critical incidents detected</div>
                        ) : (
                          alerts.map(alert => (
                            <div key={alert.id} className="mb-3 p-3 bg-white/5 border-l-2 border-brand rounded-r-lg animate-slide-in">
                               <div className="flex justify-between text-[10px] font-bold mb-1">
                                 <span className={alert.type === 'FRAG_CRITICAL' ? 'text-accent-red' : 'text-brand'}>{alert.type}</span>
                                 <span className="text-dim">{alert.time}</span>
                               </div>
                               <div className="text-xs text-secondary">{alert.msg}</div>
                            </div>
                          ))
                        )}
                      </div>
                   </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </>
  );
}

export default App;
