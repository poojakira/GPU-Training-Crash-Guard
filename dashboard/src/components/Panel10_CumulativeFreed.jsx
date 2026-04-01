import React, { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export const Panel10_CumulativeFreed = ({ totalFreedMB, history }) => {
    // Generate a cumulative trend based on the history log
    const data = useMemo(() => {
        let cumulative = 0;
        const pts = [];
        
        // Base starting point
        pts.push({ time: '0', freed: 0 });
        
        if (history && history.length > 0) {
            history.forEach((h, i) => {
                cumulative += parseFloat(h.freed || 0);
                pts.push({ 
                    time: h.timestamp || `${i}`, 
                    freed: cumulative 
                });
            });
        }
        
        // Final point matching the live state
        pts.push({ time: 'LIVE', freed: totalFreedMB || cumulative });
        
        return pts;
    }, [totalFreedMB, history]);

    return (
        <div className="hw-panel h-full w-full">
            <div className="hw-panel-header">
                <span className="panel-title">10/CUMULATIVE_RECOVERY_METRIC</span>
                <span className="text-green uppercase font-bold tracking-widest text-[9px]">Success Trace</span>
            </div>
            
            <div className="flex-1 w-full h-[300px] mt-4 font-mono text-[9px]">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                        <defs>
                            <linearGradient id="freedCol" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="var(--hw-green)" stopOpacity={0.4}/>
                                <stop offset="95%" stopColor="var(--hw-green)" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <XAxis dataKey="time" stroke="var(--text-dim)" tick={{fill: 'var(--text-dim)'}} tickLine={false} axisLine={false} />
                        <YAxis stroke="var(--text-dim)" tick={{fill: 'var(--text-dim)'}} tickLine={false} axisLine={false} />
                        <Tooltip 
                            contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--glass-border)', fontFamily: 'JetBrains Mono, monospace', fontSize: '10px' }}
                            itemStyle={{ color: 'var(--hw-green)' }}
                        />
                        <Area type="monotone" dataKey="freed" stroke="var(--hw-green)" strokeWidth={2} fillOpacity={1} fill="url(#freedCol)" isAnimationActive={true} />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
            
            <div className="mt-4 flex justify-between items-center text-dim text-[10px] mono-metric">
                <span>TOTAL_MEMORY_SAVED: <span className="text-green">{parseFloat(totalFreedMB || 0).toFixed(1)} MB</span></span>
                <span className="blink">TRACING_SUCCESS_CURVE...</span>
            </div>
        </div>
    );
};
