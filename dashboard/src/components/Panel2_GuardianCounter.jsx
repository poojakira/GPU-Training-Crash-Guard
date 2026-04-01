import React from 'react';

export const Panel2_GuardianCounter = ({ totalPrevented, msRecovered }) => {
  // Safe numeric coercion — prevents TypeError if prop arrives as string or undefined
  const count = Number(totalPrevented) || 0;
  const recovered = Number(msRecovered) || 0;

  return (
    <div className="hw-panel justify-center items-center h-full">
      <div className="hw-panel-header w-full absolute top-4 left-4 right-4">
        <span className="panel-title">02/GUARDIAN_INTERCESSIONS</span>
        <span className="text-dim">CUMULATIVE</span>
      </div>
      <div className="flex flex-col items-center mt-6">
        <div className="text-dim mb-2 uppercase tracking-widest text-[8px]">OOM Crashes Prevented</div>
        <div className="guardian-hero text-green blink mb-4">{count.toString().padStart(4, '0')}</div>
        
        <div className="flex gap-4 border-t border-dashed border-panel-border pt-4 w-full justify-center">
           <div className="text-center">
              <div className="text-dim text-[8px] uppercase">VRAM Recovered</div>
              <div className="text-amber font-bold text-2xl">+{recovered.toFixed(0)} MB</div>
           </div>
        </div>
      </div>
    </div>
  );
};
