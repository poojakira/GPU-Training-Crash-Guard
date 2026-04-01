import React, { useMemo, useEffect, useState } from 'react';

export const Panel4_AttentionHeatmap = () => {
    const [tick, setTick] = useState(0);
    
    useEffect(() => {
        const interval = setInterval(() => setTick(t => t + 1), 600);
        return () => clearInterval(interval);
    }, []);

    // Generate 8 layers x 64 traces mock attention
    const layers = useMemo(() => Array.from({length: 8}), [tick]);

    return (
        <div className="hw-panel h-full">
            <div className="hw-panel-header">
                <span className="panel-title">04/SCHEDULER_ATTN_HEATMAP</span>
                <span className="text-dim">L_1..8 / T_-64..0</span>
            </div>
            <div className="flex-1 mt-2 attn-matrix">
                <div className="flex justify-between text-[8px] text-dim">
                    <span>-6400ms</span>
                    <span>T_0</span>
                </div>
                {layers.map((_, i) => (
                    <div key={i} className="attn-row">
                        {Array.from({length: 64}).map((_, j) => {
                            // Synthesize attention wave
                            // High attention on recent blocks + random peaks
                            const noise = Math.random();
                            const recency = Math.pow(j / 64, 3); 
                            const attn = Math.max(0, recency * 0.8 + noise * 0.3);
                            
                            let color = 'rgba(255,255,255,0.02)';
                            if (attn > 0.8) color = 'var(--hw-red)';
                            else if (attn > 0.6) color = 'var(--hw-amber)';
                            else if (attn > 0.4) color = 'var(--hw-green-dim)';
                            else if (attn > 0.2) color = 'rgba(255,255,255,0.1)';

                            return <div key={j} className="attn-cell" style={{ background: color }}></div>
                        })}
                    </div>
                ))}
            </div>
        </div>
    );
};
