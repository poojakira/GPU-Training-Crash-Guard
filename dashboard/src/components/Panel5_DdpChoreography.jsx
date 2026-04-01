import React from 'react';

export const Panel5_DdpChoreography = ({ isCompacting }) => {
    return (
        <div className="hw-panel h-full">
            <div className="hw-panel-header">
               <span className="panel-title">05/DDP_BARRIER_SYNC</span>
               <span className="text-dim">ALL_REDUCE.WAIT()</span>
            </div>
            <div className="flex flex-col gap-3 flex-1 justify-center mt-2 relative">
                {[0, 1, 2, 3].map(rank => {
                    const StateLabel = isCompacting ? (rank === 0 ? "KERNEL_EXEC" : "BARRIER_WAIT") : "COMPUTE_FWD";
                    const laneColor = isCompacting ? (rank === 0 ? "bg-amber" : "bg-red fast-blink w-2 h-2 rounded-full absolute right-4") : "bg-green";

                    return (
                        <div key={rank} className="relative">
                            <div className="text-[9px] uppercase font-bold text-dim mb-1">
                                GPU_RANK_{rank} [{StateLabel}]
                            </div>
                            <div className="ddp-lane">
                                {isCompacting ? (
                                   rank === 0 ? (
                                       <div className="absolute top-0 bottom-0 left-0 right-[20%] bg-amber opacity-30"></div>
                                   ) : (
                                       <div className="absolute top-0 bottom-0 left-0 right-[20%] strip-bg border-b border-t border-amber opacity-20 relative" style={{background: 'repeating-linear-gradient(45deg, transparent, transparent 5px, rgba(255,176,0,0.1) 5px, rgba(255,176,0,0.1) 10px)'}}>
                                          <div className="w-2 h-2 bg-red rounded-full absolute right-4 top-1.5 fast-blink"></div>
                                       </div>
                                   )
                                ) : (
                                    <div className="absolute top-0 bottom-0 left-0 right-0 bg-green opacity-20"></div>
                                )}
                                {!isCompacting && <div className="w-1 h-full bg-green absolute right-0 blink shadow-[0_0_10px_#00ff2a]"></div>}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};
