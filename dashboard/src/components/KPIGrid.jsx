import React from 'react';
import { Activity, Clock, Server, ShieldCheck, AlertTriangle, Zap } from 'lucide-react';

const KPICard = ({ title, value, subtext, icon: Icon, trend, color, delayClass }) => {
  return (
    <div className={`glass-card p-6 animate-fade-in ${delayClass} group`}>
      <div className="flex justify-between items-start mb-6">
        <div className="p-3 rounded-2xl bg-white/5 border border-white/5 transition-all group-hover:border-white/10" style={{ color: color }}>
          <Icon size={24} className="drop-shadow-[0_0_8px_currentColor]" />
        </div>
        {trend && (
          <div className={`text-[10px] font-bold px-2.5 py-1 rounded-full border ${trend.isPositive ? 'bg-accent-green/10 text-accent-green border-accent-green/20' : 'bg-accent-red/10 text-accent-red border-accent-red/20'}`}>
            {trend.value}
          </div>
        )}
      </div>
      <h3 className="kpi-title mb-2 opacity-60 transition-opacity group-hover:opacity-100">{title}</h3>
      <div className="kpi-value text-4xl mb-2">{value}</div>
      <p className="text-text-dim text-[10px] font-mono tracking-wider uppercase">{subtext}</p>
      
      {/* HUD Decorative Element */}
      <div className="absolute top-2 right-2 flex gap-1 transform scale-75 opacity-20">
         <div className="w-1 h-1 bg-white rounded-full"/>
         <div className="w-1 h-1 bg-white rounded-full"/>
      </div>
    </div>
  );
};

export const KPIGrid = ({ metrics }) => {
  const OOMIcon = metrics.defragOoms === 0 ? ShieldCheck : AlertTriangle;
  
  return (
    <div className="grid grid-cols-4 gap-6 mb-12 mt-4">
      <KPICard
        title="Avg Iteration"
        value={`${metrics.defragAvgTime}s`}
        subtext={`Baseline: ${metrics.baselineAvgTime}s (REF)`}
        icon={Clock}
        trend={{ value: `-${metrics.timeReduction}%`, isPositive: true }}
        color="var(--accent-cyan)"
        delayClass="delay-100"
      />
      <KPICard
        title="Compactions"
        value={metrics.compactionEvents}
        subtext={`Salvaged: ${metrics.freedMemory} MB`}
        icon={Zap}
        color="var(--brand)"
        delayClass="delay-200"
      />
      <KPICard
        title="Peak Memory"
        value={`${metrics.defragPeak}MB`}
        subtext={`Delta: ${metrics.peakReduction}%`}
        icon={Server}
        trend={{ value: `${metrics.peakReduction > 0 ? '-' : '+'}${Math.abs(metrics.peakReduction)}%`, isPositive: metrics.peakReduction > 0 }}
        color="var(--accent-purple)"
        delayClass="delay-300"
      />
      <KPICard
        title="OOM Errors"
        value={metrics.defragOoms}
        subtext="Stability Buffer High"
        icon={OOMIcon}
        trend={metrics.defragOoms < metrics.baselineOoms ? { value: 'OPTIMAL', isPositive: true } : null}
        color="var(--accent-green)"
        delayClass="delay-100"
      />
    </div>
  );
};
