
import React from 'react';

const ModelDashboard: React.FC = () => {
  const models = [
    {
      title: 'Match Outcome Prediction',
      accuracy: 92,
      desc: 'Real-time win probability forecasting based on team compositions, early objectives, and gold differentials.',
      icon: 'videogame_asset',
      color: 'primary',
      chart: [20, 40, 30, 80, 50, 65]
    },
    {
      title: 'Rank Classification',
      accuracy: 88,
      desc: 'Deep learning analysis of player mechanics and decision patterns to identify true skill brackets versus displayed rank.',
      icon: 'military_tech',
      color: 'hextech-cyan',
      chart: [30, 45, 60, 90, 70, 75]
    },
    {
      title: 'Player Progression',
      accuracy: 85,
      desc: 'Time-series forecasting that identifies long-term improvement trends and burnout risk using historical match data.',
      icon: 'insights',
      color: 'slate-400',
      chart: [25, 35, 50, 40, 55, 30]
    },
    {
      title: 'Smurf Detection',
      accuracy: 95,
      desc: 'Identifying anomalous performance patterns and behavior inconsistent with account level or historical skill ceiling.',
      icon: 'security',
      color: 'red-400',
      chart: [40, 30, 50, 25, 75, 20]
    }
  ];

  return (
    <div className="p-8 space-y-12 max-w-[1600px] mx-auto">
      <header>
        <h1 className="text-4xl font-black text-white mb-2">MODEL DASHBOARD</h1>
        <p className="text-white/50">Real-time machine learning analytics and competitive performance tracking.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Model Cards */}
        <div className="lg:col-span-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          {models.map((model, i) => (
            <div key={i} className="glass-panel p-6 rounded-xl hover:border-primary/50 transition-all group flex flex-col">
              <div className="flex justify-between items-start mb-6">
                <div className="size-12 rounded bg-white/5 flex items-center justify-center text-primary border border-white/10">
                  <span className="material-symbols-outlined">{model.icon}</span>
                </div>
                <div className="text-right">
                  <p className="text-3xl font-black text-white">{model.accuracy}%</p>
                  <p className="text-[10px] font-bold text-white/40 uppercase tracking-widest">Accuracy</p>
                </div>
              </div>
              <h3 className="text-xl font-bold text-white mb-3">{model.title}</h3>
              <p className="text-sm text-white/50 mb-6 flex-1">{model.desc}</p>
              
              <div className="h-10 flex items-end gap-1 mb-6">
                {model.chart.map((val, idx) => (
                  <div 
                    key={idx} 
                    className={`flex-1 rounded-t-sm transition-all bg-${model.color === 'primary' ? 'primary' : (model.color === 'hextech-cyan' ? 'hextech-cyan' : (model.color === 'red-400' ? 'red-500' : 'slate-500'))} opacity-40 group-hover:opacity-100`}
                    style={{ height: `${val}%` }}
                  ></div>
                ))}
              </div>
              
              <button className="w-full py-2.5 bg-white/5 group-hover:bg-primary group-hover:text-background-dark rounded font-bold text-sm transition-all flex items-center justify-center gap-2">
                <span className="material-symbols-outlined text-sm">rocket_launch</span>
                LAUNCH MODEL
              </button>
            </div>
          ))}
        </div>

        {/* System Health / Sidebar */}
        <div className="lg:col-span-4 space-y-6">
          <div className="glass-panel p-6 rounded-xl">
            <h3 className="text-white font-bold mb-6 flex items-center gap-2 uppercase tracking-widest text-sm">
              <span className="material-symbols-outlined text-primary">analytics</span>
              System Health
            </h3>
            <div className="space-y-6">
              {[
                { label: 'Predictive Engine', status: 'STABLE', color: 'text-green-500', bar: 95 },
                { label: 'GPU Cluster Alpha', status: 'OPTIMAL', color: 'text-green-500', bar: 100 },
                { label: 'Smurf Detection Service', status: 'TRAINING', color: 'text-yellow-500', bar: 65 },
                { label: 'Global Data Ingest', status: 'ACTIVE', color: 'text-green-500', bar: 85 },
              ].map((sys, i) => (
                <div key={i}>
                  <div className="flex justify-between items-center mb-2">
                    <p className="text-xs font-bold text-white/60">{sys.label}</p>
                    <p className={`text-[10px] font-black ${sys.color}`}>{sys.status}</p>
                  </div>
                  <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${sys.status === 'TRAINING' ? 'bg-yellow-500' : 'bg-green-500'} transition-all`}
                      style={{ width: `${sys.bar}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-8 pt-8 border-t border-white/10">
              <div className="flex justify-between mb-2">
                <p className="text-[10px] font-bold text-white/40 uppercase">Storage Usage</p>
                <p className="text-[10px] font-bold text-white">4.2 TB / 10 TB</p>
              </div>
              <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden flex">
                <div className="h-full bg-primary/40 w-[30%]"></div>
                <div className="h-full bg-hextech-blue w-[12%]"></div>
              </div>
            </div>
          </div>

          <div className="p-6 rounded-xl bg-primary/10 border border-primary/20 relative overflow-hidden group">
            <div className="absolute -top-10 -right-10 size-40 bg-primary/5 blur-[40px] rounded-full group-hover:bg-primary/10 transition-all"></div>
            <div className="relative z-10">
              <div className="flex items-center gap-2 mb-4">
                <span className="material-symbols-outlined text-primary">auto_awesome</span>
                <p className="text-sm font-black text-white">New Patch v14.2</p>
              </div>
              <p className="text-xs text-white/60 leading-relaxed mb-4">
                Neural networks are currently re-weighting objective values based on the latest map changes. Model retraining will finish in 4h.
              </p>
              <button className="text-primary text-[10px] font-bold uppercase tracking-widest flex items-center gap-1 hover:gap-2 transition-all">
                Read patch notes <span className="material-symbols-outlined text-sm">arrow_forward</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelDashboard;
