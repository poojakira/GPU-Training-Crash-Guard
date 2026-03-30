import React from 'react';

export const MemoryMap = ({ fragPercent, fullSize }) => {
  const totalBlocks = fullSize ? 800 : 200;
  const fragmentedBlocks = Math.round((fragPercent / 100) * totalBlocks);
  
  const blocks = [];
  for (let i = 0; i < totalBlocks; i++) {
    const scatter = (i * 137) % totalBlocks;
    const isFragmented = scatter < fragmentedBlocks;
    const isGlow = i % 47 === 0; // Subtle random pulse

    blocks.push(
      <div 
        key={i} 
        className={`memory-block ${isFragmented ? 'warning' : 'active'} ${isGlow ? 'glow' : ''}`}
      />
    );
  }

  return (
    <div className={`p-0 flex flex-col h-full bg-transparent overflow-hidden`}>
      <div className="flex justify-between items-center mb-4 px-1">
        <div className="flex flex-col">
           <span className="kpi-title mb-0.5" style={{fontSize: '0.6rem'}}>Physical Mapping</span>
           <span className="text-[10px] font-mono text-dim uppercase tracking-wider">{totalBlocks} Pages @ 4KB</span>
        </div>
        <div className="flex gap-4">
           <div className="flex items-center gap-1.5">
             <div className="w-1.5 h-1.5 rounded-full bg-accent-green" />
             <span className="text-[9px] font-bold text-dim uppercase">Alloc</span>
           </div>
           <div className="flex items-center gap-1.5">
             <div className="w-1.5 h-1.5 rounded-full bg-accent-red" />
             <span className="text-[9px] font-bold text-dim uppercase">Frag</span>
           </div>
        </div>
      </div>
      
      <div className={`memory-grid flex-1 border border-white/5 p-2 rounded-lg bg-black/20 ${fullSize ? 'large' : 'small'}`}>
        {blocks}
      </div>

      {/* Footer HUD elements */}
      <div className="mt-4 flex justify-between items-center px-1">
         <div className="text-[9px] font-mono text-dim tracking-tighter">ADDR: 0x0000 - 0x7FFF</div>
         <div className="flex gap-2">
            <div className="w-10 h-[1px] bg-white/10"/>
            <div className="w-10 h-[1px] bg-brand"/>
         </div>
      </div>
    </div>
  );
};

