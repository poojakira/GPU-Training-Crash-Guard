import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export const Panel11_AllocationDist = ({ fragPercent }) => {
    // Generate a distribution based on the fragmentation percentage
    const data = useMemo(() => {
        const live = 100 - fragPercent;
        const holes = fragPercent * 0.7; // Estimate portion of holes within frag %
        const free = 100 - live - holes;
        
        return [{
            name: 'VRAM_STATES',
            LIVE_TENSORS: live,
            FRAG_HOLES: holes,
            CONTIGUOUS_FREE: free
        }];
    }, [fragPercent]);

    return (
        <div className="hw-panel h-full w-full">
            <div className="hw-panel-header">
                <span className="panel-title">11/ALLOCATION_DISTRIBUTION_PROFILE</span>
                <span className="text-dim uppercase font-bold tracking-widest text-[9px]">Structural breakdown</span>
            </div>
            
            <div className="flex-1 w-full h-[500px] mt-4 font-mono text-[9px]">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart layout="vertical" data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                        <XAxis type="number" stroke="var(--text-dim)" domain={[0, 100]} hide />
                        <YAxis type="category" dataKey="name" hide />
                        <Tooltip 
                            contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--glass-border)', fontFamily: 'JetBrains Mono, monospace', fontSize: '10px' }}
                            cursor={{ fill: 'transparent' }}
                        />
                        <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px', fontFamily: 'JetBrains Mono' }} />
                        <Bar dataKey="LIVE_TENSORS" stackId="a" fill="var(--hw-green)" />
                        <Bar dataKey="FRAG_HOLES" stackId="a" fill="var(--hw-red)" />
                        <Bar dataKey="CONTIGUOUS_FREE" stackId="a" fill="transparent" stroke="var(--glass-border)" strokeWidth={1} style={{ opacity: 0.3 }} />
                    </BarChart>
                </ResponsiveContainer>
            </div>
            
            <div className="mt-4 grid grid-cols-3 gap-4 text-center mono-metric text-[10px]">
                <div className="flex flex-col gap-1">
                    <span className="text-dim uppercase tracking-widest">LIVE_ALLOC</span>
                    <span className="text-xl font-bold text-green">{(100 - fragPercent).toFixed(1)}%</span>
                </div>
                <div className="flex flex-col gap-1">
                    <span className="text-dim uppercase tracking-widest">FRAG_LOSS</span>
                    <span className="text-xl font-bold text-red">{fragPercent.toFixed(1)}%</span>
                </div>
                <div className="flex flex-col gap-1">
                    <span className="text-dim uppercase tracking-widest">COLLECTIBLE</span>
                    <span className="text-xl font-bold text-amber">{(fragPercent * 0.4).toFixed(1)}%</span>
                </div>
            </div>
        </div>
    );
};
