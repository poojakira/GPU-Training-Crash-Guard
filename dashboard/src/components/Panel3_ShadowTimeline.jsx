import React from 'react';

export const Panel3_ShadowTimeline = ({ currentFrag, thresholdLevel }) => {
  // Synthesize prediction 100ms ahead
  // If currentFrag is rising quickly, prediction is much higher
  const predictionFrag = Math.min(100, currentFrag * 1.3 + Math.random() * 5);
  
  const isDanger = predictionFrag > thresholdLevel;
  const isCritical = currentFrag > thresholdLevel;

  return (
    <div className="hw-panel justify-between h-full">
      <div className="hw-panel-header">
        <span className="panel-title">03/PREDICTIVE_SHADOW_TIMELINE</span>
        <span className="text-dim">FRAG_PCT (T + 100ms)</span>
      </div>
      
      <div className="flex flex-col gap-4 mt-4 relative">
        <div className="flex justify-between text-[10px] uppercase font-bold text-dim mb-[-5px]">
          <span>GPU 0_RANK (LEAD)</span>
          <span className={isDanger ? "text-amber fast-blink" : "text-green"}>
              {isDanger ? 'WARN_LIMIT_REACHED' : 'NOMINAL'}
          </span>
        </div>
        
        <div className="timeline-track w-full">
            {/* The ghost predicted timeline */}
            <div 
               className="timeline-ghost" 
               style={{ 
                   width: `${predictionFrag}%`, 
                   background: isDanger ? 'repeating-linear-gradient(45deg, transparent, transparent 2px, var(--hw-red) 2px, var(--hw-red) 4px)' : undefined 
               }}
            ></div>
            
            {/* The solid current timeline */}
            <div 
               className={`timeline-fill ${isCritical ? 'bg-red' : 'bg-green'}`} 
               style={{ width: `${currentFrag}%` }}
            ></div>
            
            {/* The hard threshold marker */}
            <div 
               className="absolute top-[-5px] bottom-[-5px] w-0.5 bg-red z-10"
               style={{ left: `${thresholdLevel}%` }}
            >
                <div className="absolute top-[-15px] left-[-20px] text-red text-[8px]">TH_80</div>
            </div>
        </div>

        <div className="flex justify-between items-end mt-4">
           <div>
               <div className="text-[10px] text-dim">CURRENT_STATE</div>
               <div className="text-xl font-bold">{currentFrag.toFixed(1)}%</div>
           </div>
           <div className="text-right">
               <div className="text-[10px] text-dim blink">T+100ms FORECAST</div>
               <div className={`text-xl font-bold ${isDanger ? 'text-amber' : ''}`}>{predictionFrag.toFixed(1)}%</div>
           </div>
        </div>
      </div>
    </div>
  );
};
