import React, { useMemo } from 'react';

export const Panel7_CompactionRay = ({ isCompacting }) => {
   // Generate two tracks of memory: Before and After
   // Before has holes. After is densely packed.
   
   const numBlocks = 64;
   
   const { source, dest } = useMemo(() => {
       const src = [];
       const dst = [];
       
       let filledCount = 0;
       for(let i=0; i<numBlocks; i++) {
           const isEmpty = Math.random() > 0.6;
           if (isEmpty) {
               src.push({ id: i, active: false });
           } else {
               src.push({ id: i, active: true });
               dst.push({ id: filledCount, active: true });
               filledCount++;
           }
       }
       // Fill the rest of dst with empty
       for(let i=filledCount; i<numBlocks; i++) {
           dst.push({ id: i, active: false });
       }
       
       return { source: src, dest: dst };
   }, []);

   return (
       <div className="hw-panel h-[180px]">
           <div className="hw-panel-header">
               <span className="panel-title">07/PHYSICAL_COMPACTION_RAY</span>
               <span className="text-dim">TRITON_COPY_REPACK</span>
           </div>
           
           <div className="flex flex-col flex-1 justify-around mt-4">
               {/* Source Track */}
               <div className="relative">
                   <div className="text-[8px] text-dim mb-1 tracking-widest uppercase">PRE-SWEEP_ADDRESS_SPACE</div>
                   <div className="compaction-track opacity-80">
                      {source.map(b => (
                         <div key={b.id} className="flex-1 h-full border-r border-panel border-opacity-10" 
                              style={{ background: b.active ? (isCompacting ? 'var(--hw-amber)' : 'var(--text-dim)') : 'rgba(255,42,42,0.1)' }}>
                         </div>
                      ))}
                   </div>
               </div>
               
               {/* Ray/diagonal section */}
               <div className="h-6 w-full relative overflow-hidden flex items-center">
                   {isCompacting && (
                       <div className="w-full text-center text-amber text-[10px] uppercase fast-blink">
                           &gt;&gt; RELOCATING BASE POINTERS &lt;&lt;
                       </div>
                   )}
                   {/* In a real DOM setup SVG would draw the lines. Here we imply it purely with visual space and amber flashing */}
               </div>

               {/* Target Track */}
               <div className="relative">
                   <div className="text-[8px] text-green mb-1 tracking-widest uppercase">CONTIGUOUS_POST_SWEEP</div>
                   <div className="compaction-track">
                      {dest.map(b => (
                         <div key={b.id} className="flex-1 h-full border-r border-bg-core" 
                              style={{ background: b.active ? (isCompacting ? 'var(--hw-amber)' : 'var(--hw-green)') : 'transparent' }}>
                         </div>
                      ))}
                   </div>
               </div>
           </div>
       </div>
   );
};
