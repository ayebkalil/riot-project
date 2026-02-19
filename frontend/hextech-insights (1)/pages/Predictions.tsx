
import React from 'react';
import Gauge from '../components/Gauge';

const Predictions: React.FC = () => {
  return (
    <div className="p-8 max-w-[1600px] mx-auto">
       <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-white text-4xl font-black leading-tight tracking-tight">Outcome Prediction</h1>
          <p className="text-white/50 text-base">Advanced algorithmic forecasting based on current meta and player history.</p>
        </div>
        <div className="flex gap-3">
          <button className="flex items-center gap-2 px-6 py-2.5 bg-white/5 hover:bg-white/10 text-white rounded-lg font-bold text-sm transition-all border border-white/10">
            <span className="material-symbols-outlined text-lg">sync</span>
            Sync Live Match
          </button>
          <button className="px-8 py-2.5 bg-primary hover:bg-primary/80 text-background-dark rounded-lg font-bold text-sm transition-all shadow-[0_0_20px_rgba(200,170,111,0.3)]">
            Run Analysis
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-7 space-y-6">
          {/* Team Selection */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Team A */}
            <div className="glass-panel p-5 rounded-xl border-l-4 border-l-blue-500">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-blue-400 font-bold flex items-center gap-2">
                  <span className="material-symbols-outlined text-lg">shield</span>
                  TEAM A (BLUE)
                </h3>
                <span className="text-[10px] bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded font-bold uppercase tracking-wider">Home</span>
              </div>
              <div className="space-y-4">
                {[
                  { role: 'Top Lane', champ: 'Aatrox (KDA: 3.2)', progress: 75, img: 'https://picsum.photos/seed/aatrox/100' },
                  { role: 'Jungle', champ: 'Lee Sin (KDA: 4.1)', progress: 50, img: 'https://picsum.photos/seed/leesin/100' },
                ].map((slot, i) => (
                  <div key={i} className="flex items-center gap-4 group cursor-pointer">
                    <div className="size-12 rounded-lg bg-white/5 border border-white/10 overflow-hidden relative">
                      <div className="absolute inset-0 bg-cover bg-center" style={{ backgroundImage: `url(${slot.img})` }}></div>
                    </div>
                    <div className="flex-1">
                      <p className="text-[10px] text-white/40 font-bold uppercase">{slot.role}</p>
                      <p className="text-sm font-semibold text-white">{slot.champ}</p>
                    </div>
                    <div className="h-1.5 w-12 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full bg-primary" style={{ width: `${slot.progress}%` }}></div>
                    </div>
                  </div>
                ))}
                <div className="flex items-center gap-4 opacity-60">
                  <div className="size-12 rounded-lg bg-white/5 border border-dashed border-white/20 flex items-center justify-center text-white/20">
                    <span className="material-symbols-outlined">add</span>
                  </div>
                  <p className="text-xs text-white/30 italic">Assign Remaining Slots...</p>
                </div>
              </div>
            </div>

            {/* Team B */}
            <div className="glass-panel p-5 rounded-xl border-l-4 border-l-red-500">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-red-400 font-bold flex items-center gap-2">
                  <span className="material-symbols-outlined text-lg">swords</span>
                  TEAM B (RED)
                </h3>
                <span className="text-[10px] bg-red-500/20 text-red-400 px-2 py-0.5 rounded font-bold uppercase tracking-wider">Away</span>
              </div>
              <div className="space-y-4">
                {[
                  { role: 'Mid Lane', champ: 'Azir (KDA: 2.8)', progress: 80, img: 'https://picsum.photos/seed/azir/100' },
                  { role: 'ADC', champ: 'Jinx (KDA: 5.5)', progress: 66, img: 'https://picsum.photos/seed/jinx/100' },
                ].map((slot, i) => (
                  <div key={i} className="flex items-center gap-4 group cursor-pointer">
                    <div className="size-12 rounded-lg bg-white/5 border border-white/10 overflow-hidden relative">
                      <div className="absolute inset-0 bg-cover bg-center" style={{ backgroundImage: `url(${slot.img})` }}></div>
                    </div>
                    <div className="flex-1">
                      <p className="text-[10px] text-white/40 font-bold uppercase">{slot.role}</p>
                      <p className="text-sm font-semibold text-white">{slot.champ}</p>
                    </div>
                    <div className="h-1.5 w-12 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full bg-hextech-cyan" style={{ width: `${slot.progress}%` }}></div>
                    </div>
                  </div>
                ))}
                 <div className="flex items-center gap-4 opacity-60">
                  <div className="size-12 rounded-lg bg-white/5 border border-dashed border-white/20 flex items-center justify-center text-white/20">
                    <span className="material-symbols-outlined">add</span>
                  </div>
                  <p className="text-xs text-white/30 italic">Assign Remaining Slots...</p>
                </div>
              </div>
            </div>
          </div>

          {/* Performance Toggles */}
          <div className="glass-panel p-6 rounded-xl">
            <h3 className="text-white font-bold mb-6 flex items-center gap-2">
              <span className="material-symbols-outlined text-primary">analytics</span>
              Performance Metrics
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-6">
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-xs font-medium text-white/60">Average Gold/Min (Team A)</label>
                    <span className="text-xs font-bold text-primary">1,850g</span>
                  </div>
                  <input className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary" type="range" defaultValue="75" />
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-xs font-medium text-white/60">Vision Score Index</label>
                    <span className="text-xs font-bold text-primary">78%</span>
                  </div>
                  <input className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary" type="range" defaultValue="78" />
                </div>
              </div>
              <div className="space-y-3">
                {[
                  { label: 'Current Patch Synergy (v13.24)', active: true },
                  { label: 'Recent Win Streak Bonus', active: false },
                  { label: 'Late Game Scaling Focus', active: true },
                ].map((toggle, i) => (
                  <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/10">
                    <span className="text-xs font-medium">{toggle.label}</span>
                    <div className={`w-10 h-5 rounded-full relative cursor-pointer transition-all ${toggle.active ? 'bg-primary/30' : 'bg-white/10'}`}>
                      <div className={`absolute top-1 size-3 rounded-full transition-all ${toggle.active ? 'right-1 bg-primary' : 'left-1 bg-white/40'}`}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel: Results */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <div className="glass-panel p-8 rounded-2xl flex flex-col items-center relative overflow-hidden border-t-2 border-t-primary/50">
             <div className="absolute top-0 right-0 size-32 bg-primary/5 blur-[80px]"></div>
             <h3 className="text-white/40 text-xs font-black uppercase tracking-[0.2em] mb-8">Win Probability</h3>
             <Gauge value={68} label="Team A Advantage" />
             <div className="bg-primary/20 text-primary px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest border border-primary/30 my-8">
                High Confidence Analysis
             </div>
             <div className="w-full space-y-4">
                <p className="text-white/50 text-[11px] font-bold uppercase tracking-wider mb-2">Key Influence Factors</p>
                {[
                  { label: 'Team Comp Scaling', sub: 'Team A excels past 30:00', val: '+15.2%', up: true, icon: 'bolt', color: 'primary' },
                  { label: 'Objective Control', sub: 'Team B superior dragon secure', val: '-5.8%', up: false, icon: 'flag', color: 'hextech-cyan' },
                  { label: 'Lane Dominance', sub: 'Mid lane matchup advantage', val: '+8.4%', up: true, icon: 'psychology', color: 'primary' },
                ].map((factor, i) => (
                  <div key={i} className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10 hover:border-primary/30 transition-all cursor-default">
                    <div className="flex items-center gap-3">
                      <div className={`size-8 rounded bg-${factor.color}/10 flex items-center justify-center text-${factor.color}`}>
                        <span className="material-symbols-outlined text-lg">{factor.icon}</span>
                      </div>
                      <div>
                        <p className="text-sm font-bold text-white">{factor.label}</p>
                        <p className="text-[10px] text-white/40">{factor.sub}</p>
                      </div>
                    </div>
                    <span className={`text-sm font-bold ${factor.up ? 'text-green-400' : 'text-red-400'}`}>{factor.val}</span>
                  </div>
                ))}
             </div>
          </div>

          <div className="p-6 rounded-xl bg-primary/10 border border-primary/20">
            <div className="flex gap-4">
              <span className="material-symbols-outlined text-primary">info</span>
              <div>
                <p className="text-sm font-bold text-primary mb-1">Analyst Note</p>
                <p className="text-xs text-white/70 leading-relaxed">
                  Team A's composition has a statistically higher win rate against Team B's front line in late-game sieges. Expect the win probability to peak at the 32-minute mark if gold parity is maintained.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Predictions;
