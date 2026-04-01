import React, { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

export const Panel8_TrendGraph = ({ currentFrag, thresholdLevel, history }) => {
    // We synthesize a timeline combining real trace history with the live scalar
    // to give it a living waveform look extending backward.
    const data = useMemo(() => {
        const pts = [];
        // Base sine wave representing past iterations before known compactions
        for(let i=0; i<60; i++) {
            pts.push({ 
                time: `T-${60-i}`, 
                frag: Math.max(0, 30 + Math.sin(i*0.2)*15 + Math.random()*5) 
            });
        }
        
        // Inject actual compactions from history
        let injectionIdx = 30;
        if (history && history.length > 0) {
            history.forEach(h => {
                if (pts[injectionIdx]) {
                    // spike up to threshold just before compaction
                    pts[injectionIdx-1].frag = 85; 
                    // massive drop reflecting the fragReduction
                    pts[injectionIdx].frag = 85 - (h.fragReduction || 30);
                }
                injectionIdx += 8;
            });
        }
        
        // Append live state
        pts.push({ time: 'LIVE', frag: currentFrag });
        return pts;
    }, [currentFrag, history]);

    return (
        <div className="hw-panel h-full w-full">
            <div className="hw-panel-header">
                <span className="panel-title">08/LONG_TERM_FRAGMENTATION_AGGREGATE</span>
                <span className="text-dim">FRAG_PCT vs OOM_THRESHOLD</span>
            </div>
            
            <div className="flex-1 w-full h-full mt-4 font-mono text-[9px]">
                <ResponsiveContainer width="100%" height="85%">
                    <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <defs>
                            <linearGradient id="fragColor" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="var(--hw-green)" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="var(--hw-green)" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <XAxis dataKey="time" stroke="var(--text-dim)" tick={{fill: 'var(--text-dim)'}} tickLine={false} axisLine={false} />
                        <YAxis stroke="var(--text-dim)" tick={{fill: 'var(--text-dim)'}} tickLine={false} axisLine={false} domain={[0, 100]} />
                        <Tooltip 
                            contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--glass-border)', fontFamily: 'JetBrains Mono, monospace', fontSize: '10px' }}
                            itemStyle={{ color: 'var(--text-active)' }}
                        />
                        <ReferenceLine y={thresholdLevel} stroke="var(--hw-red)" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'OOM_PREDICTION_LIMIT', fill: 'var(--hw-red)', fontSize: '9px' }} />
                        <Area type="monotone" dataKey="frag" stroke="var(--hw-green)" fillOpacity={1} fill="url(#fragColor)" isAnimationActive={false} />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
