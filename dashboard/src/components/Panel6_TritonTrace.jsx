import React, { useEffect, useState } from 'react';

export const Panel6_TritonTrace = ({ isCompacting, lastCompactionTime }) => {
    const [logs, setLogs] = useState([
        { id: 1, time: '14.003ms', op: 'INIT_SNAPSHOT', status: 'OK' },
        { id: 2, time: '14.052ms', op: 'PREDICT_MODEL_FWD', status: 'OK' },
    ]);

    useEffect(() => {
        if (isCompacting) {
            setLogs(prev => [
                ...prev.slice(-30),
                { id: Date.now(), time: `${(Date.now() % 100000)/1000}ms`, op: 'TRITON_JIT_COMPILE', status: 'WAIT' }
            ]);
            setTimeout(() => {
                setLogs(prev => [
                    ...prev.slice(-30),
                    { id: Date.now()+1, time: `${(Date.now() % 100000)/1000}ms`, op: 'DDP_SYNC_BARRIER', status: 'LOCK' },
                    { id: Date.now()+2, time: `${(Date.now() % 100000)/1000}ms`, op: 'COPY_SRC_DST', status: 'SWEEPING' },
                ]);
            }, 500);
        } else if (lastCompactionTime) {
            setLogs(prev => {
                if (prev.length > 0 && prev[prev.length-1].op !== 'BARRIER_RELEASE') {
                    return [
                        ...prev.slice(-30),
                        { id: Date.now(), time: `${(Date.now() % 100000)/1000}ms`, op: 'GC_CLEANUP', status: 'OK' },
                        { id: Date.now()+1, time: `${(Date.now() % 100000)/1000}ms`, op: 'BARRIER_RELEASE', status: 'SYNC' },
                    ];
                }
                return prev;
            });
        }
    }, [isCompacting, lastCompactionTime]);

    return (
        <div className="hw-panel h-full overflow-hidden flex flex-col">
            <div className="hw-panel-header">
                <span className="panel-title">06/TRITON_EXEC_TRACE</span>
                <span className="text-amber">LIVE_STDOUT</span>
            </div>
            <div className="flex-1 overflow-hidden relative">
                <div className="absolute inset-0 overflow-y-auto w-full flex flex-col-reverse p-1">
                    {logs.map((log) => {
                       let statusClass = 'text-green';
                       if (log.status === 'WAIT' || log.status === 'LOCK' || log.status === 'SWEEPING') statusClass = 'text-amber blink';
                       if (log.status === 'ERR') statusClass = 'text-red fast-blink';
                       
                       return (
                           <div key={log.id} className="trace-row">
                               <div className="text-dim">{log.time}</div>
                               <div className="text-white">{log.op}</div>
                               <div className={`text-right ${statusClass}`}>[{log.status}]</div>
                           </div>
                       );
                    })}
                </div>
            </div>
        </div>
    );
};
