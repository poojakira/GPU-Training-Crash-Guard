import React, { useMemo } from 'react';

export const Panel1_VramMap = ({ fragPercent, isCompacting }) => {
  // Generate 2048 cells (approx 64x32 grid layout)
  const cells = useMemo(() => {
    const count = 2048;
    const array = new Array(count).fill(0);
    return array.map((_, i) => {
      // 0: Free, 1: Live, 2: Fragmented, 3: Compacting
      const isNoise = Math.random() < 0.3; 
      const isFragment = Math.random() < (fragPercent / 100) * 0.4;
      
      let state = 0; // free/dark
      if (isNoise) state = 1; // live
      if (isFragment) state = 2; // hole
      
      return state;
    });
  }, [fragPercent]);

  return (
    <div className="hw-panel h-full">
      <div className="hw-panel-header">
        <span className="panel-title">01/PHYSICAL_VRAM_MAP // [80GB TOPOLOGY]</span>
        <span className="text-dim">RES: 39MB/BLK</span>
      </div>
      <div className="vram-grid-container flex-1 overflow-hidden">
        {cells.map((state, i) => {
          let bgClass = "bg-panel-border";
          if (state === 1) bgClass = "bg-green"; // Live
          if (state === 2) bgClass = "bg-red fast-blink"; // Fragment
          if (isCompacting && state === 2) bgClass = "bg-amber fast-blink";
          
          return (
            <div 
              key={i} 
              className={`vram-cell ${bgClass}`}
              style={{ opacity: state === 0 ? 0.2 : 0.8 }}
            />
          );
        })}
      </div>
    </div>
  );
};
