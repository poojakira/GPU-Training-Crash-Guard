import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export const Panel12_SyncOverhead = ({ isCompacting }) => {
    // Generate jittery sync latency data for DDP lanes
    const data = useMemo(() => {
        const pts = [];
        for(let i=0; i<30; i++) {
            const base = 2 + Math.random() * 3;
            // Spikes during compaction
            const spike = isCompacting ? 15 + Math.random() * 10 : 0;
            pts.push({ 
                step: i, 
                latency: base + spike 
            });
        }
        return pts;
    }, [isCompacting]);

    return (
        <div className="hw-panel h-full w-full">
            <div className="hw-panel-header">
                <span className="panel-title">12/DDP_SYNC_OVERHEAD_ANALYSIS</span>
                <span className="text-amber uppercase font-bold tracking-widest text-[9px]">NCCL_BROADCAST_LATENCY</span>
            </div>
            
            <div className="flex-1 w-full h-[400px] mt-4 font-mono text-[9px]">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 20, right: 30, left: -20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--glass-border)" vertical={false} />
                        <XAxis dataKey="step" stroke="var(--text-dim)" tick={{fill: 'var(--text-dim)'}} tickLine={false} axisLine={false} />
                        <YAxis stroke="var(--text-dim)" tick={{fill: 'var(--text-dim)'}} tickLine={false} axisLine={false} domain={[0, 40]} />
                        <Tooltip 
                            contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--glass-border)', fontFamily: 'JetBrains Mono, monospace', fontSize: '10px' }}
                            itemStyle={{ color: 'var(--hw-amber)' }}
                        />
                        <Line type="stepAfter" dataKey="latency" stroke="var(--hw-amber)" strokeWidth={2} dot={{r: 1, fill: 'var(--hw-amber)'}} activeDot={{r: 4}} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
            
            <div className="mt-4 flex flex-col gap-2">
                <div className="flex justify-between items-center text-dim text-[10px] mono-metric">
                    <span>AVG_OVERHEAD: <span className="text-amber">{(isCompacting ? 18.5 : 3.2).toFixed(2)}ms</span></span>
                    <span className="text-white uppercase font-bold">NCCL_GROUP_SYNC: {isCompacting ? 'STALLED_FOR_SWEEP' : 'ACTIVE_BROADCAST'}</span>
                </div>
                <div className="w-full h-1 bg-glass-border rounded-full overflow-hidden">
                    <div className={`h-full transition-all duration-300 ${isCompacting ? 'w-full bg-red blink' : 'w-1/4 bg-green'}`}></div>
                </div>
            </div>
        </div>
    );
};
