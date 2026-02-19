
import React, { useState, useEffect } from 'react';
import { AreaChart, Area, ResponsiveContainer, XAxis, Tooltip, CartesianGrid } from 'recharts';

const progressionData = [
  { game: 'Game 1', gain: 10 },
  { game: 'Game 2', gain: 25 },
  { game: 'Game 3', gain: 20 },
  { game: 'Game 4', gain: 45 },
  { game: 'Game 5', gain: 30 },
  { game: 'Game 6', gain: 60 },
  { game: 'Game 7', gain: 55 },
  { game: 'Game 8', gain: 80 },
  { game: 'Game 9', gain: 65 },
  { game: 'Game 10', gain: 95 },
  { game: 'Game 11', gain: 85 },
  { game: 'Game 12', gain: 110 },
  { game: 'Game 13', gain: 90 },
  { game: 'Game 14', gain: 125 },
  { game: 'Game 15', gain: 115 },
];

const detectionFactors = [
  { 
    label: 'Mechanical Outlier', 
    value: 'High', 
    color: 'text-primary', 
    icon: 'bolt',
    context: 'Skillshots land 42% more often than league average.'
  },
  { 
    label: 'APM Consistency', 
    value: '380+', 
    color: 'text-hextech-blue', 
    icon: 'speed',
    context: 'Input variance is < 2ms, typical of Pro-level hardware/mechanics.'
  },
  { 
    label: 'Pathing Efficiency', 
    value: '98%', 
    color: 'text-green-400', 
    icon: 'map',
    context: 'Jungle route optimization matches Diamond+ patterns.'
  },
  { 
    label: 'Itemization Speed', 
    value: 'Critical', 
    color: 'text-red-400', 
    icon: 'shopping_cart',
    context: 'Recall-to-buy window is 1.4s (Tier 1 speed).'
  },
];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[#0a1428] border border-primary/30 p-2 rounded shadow-lg backdrop-blur-md">
        <p className="text-[10px] font-black text-primary uppercase tracking-widest mb-1">{label}</p>
        <p className="text-sm font-bold text-white">
          Gain: <span className="text-hextech-blue">+{payload[0].value} LP</span>
        </p>
      </div>
    );
  }
  return null;
};

const Profile: React.FC = () => {
  const [pulse, setPulse] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => setPulse(p => !p), 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-8 max-w-[1400px] mx-auto space-y-8">
      {/* Profile Header */}
      <section className="glass-panel p-6 rounded-xl flex flex-col lg:flex-row lg:items-center justify-between gap-6 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-full bg-gradient-to-l from-primary/5 to-transparent pointer-events-none"></div>
        <div className="flex gap-6 items-center z-10">
          <div className="relative">
            <div 
              className="size-32 rounded-xl border-2 border-primary shadow-[0_0_15px_rgba(200,170,111,0.3)] bg-cover bg-center" 
              style={{ backgroundImage: `url('https://picsum.photos/seed/profile/200')` }}
            ></div>
            <span className="absolute -bottom-2 left-1/2 -translate-x-1/2 bg-background-dark border border-primary text-primary text-[10px] px-2 py-0.5 font-bold rounded">LVL 342</span>
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <h1 className="text-white text-3xl font-black tracking-tight">Hide on bush <span className="text-slate-500 font-medium">#KR1</span></h1>
              <span className="material-symbols-outlined text-primary text-xl animate-pulse">verified</span>
            </div>
            <div className="flex items-center gap-4 mt-1">
              <p className="text-primary font-bold text-lg">Diamond I <span className="text-slate-400 font-normal ml-1">45 LP</span></p>
              <div className="w-1 h-1 bg-slate-600 rounded-full"></div>
              <p className="text-slate-400 text-sm uppercase tracking-wider font-semibold">Ranked Solo</p>
            </div>
            <div className="flex items-center gap-4 mt-3">
              <div className="flex flex-col">
                <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Win Rate</span>
                <span className="text-hextech-blue font-bold text-lg">65.2%</span>
              </div>
              <div className="w-px h-8 bg-slate-700"></div>
              <div className="flex flex-col">
                <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">W / L</span>
                <span className="text-slate-300 font-medium text-lg">120W <span className="text-slate-500">/</span> 64L</span>
              </div>
            </div>
          </div>
        </div>
        <div className="flex gap-3 z-10">
          <button className="flex min-w-[140px] items-center justify-center rounded-lg h-12 px-6 bg-white/5 border border-primary/40 text-primary hover:bg-primary/20 transition-all text-sm font-bold uppercase tracking-widest">
            <span className="material-symbols-outlined mr-2 text-sm">refresh</span> Update
          </button>
          <button className="flex min-w-[140px] items-center justify-center rounded-lg h-12 px-6 bg-primary text-background-dark hover:bg-white transition-all text-sm font-bold uppercase tracking-widest shadow-[0_0_20px_rgba(200,170,111,0.4)]">
            Live Game
          </button>
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Sidebar Analytics */}
        <div className="space-y-6">
          {/* REFINED: Advanced Smurf Probability Visualization */}
          <div className="glass-panel p-6 rounded-xl flex flex-col relative overflow-hidden group">
            {/* Ambient Background Elements */}
            <div className={`absolute top-0 right-0 p-4 transition-opacity duration-1000 ${pulse ? 'opacity-10' : 'opacity-30'}`}>
              <span className="material-symbols-outlined text-primary text-6xl select-none">radar</span>
            </div>
            <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none"></div>

            <header className="flex justify-between items-start mb-6 relative z-10">
              <div>
                <h3 className="text-slate-200 text-sm font-black uppercase tracking-widest">Smurf Confidence</h3>
                <p className="text-[10px] text-primary/60 font-bold uppercase">Algorithm: V3.2-NEURAL</p>
              </div>
              <div className="size-2 bg-red-500 rounded-full animate-ping"></div>
            </header>
            
            <div className="relative size-56 self-center flex items-center justify-center mb-8">
               {/* Background Glow */}
               <div className="absolute inset-0 bg-primary/10 rounded-full blur-[40px] opacity-50 group-hover:opacity-80 transition-opacity"></div>
               
               {/* Advanced Multi-layered Arc Meter */}
               <svg className="size-full -rotate-180 relative z-10" viewBox="0 0 100 100">
                {/* Track */}
                <path d="M 20 80 A 40 40 0 1 1 80 80" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="6" strokeLinecap="round" />
                {/* Shimmer Effect */}
                <path 
                  d="M 20 80 A 40 40 0 1 1 80 80" 
                  fill="none" 
                  stroke="rgba(200, 170, 111, 0.1)" 
                  strokeWidth="6" 
                  strokeDasharray="188" 
                  strokeDashoffset="28" // 85%
                  strokeLinecap="round" 
                />
                {/* Main Progress Arc */}
                <path 
                  d="M 20 80 A 40 40 0 1 1 80 80" 
                  fill="none" 
                  stroke="url(#arcGradient)" 
                  strokeWidth="8" 
                  strokeDasharray="188" 
                  strokeDashoffset="28" // 85%
                  strokeLinecap="round" 
                  className="transition-all duration-[2000ms] ease-out drop-shadow-[0_0_12px_#c8aa6f]"
                />
                <defs>
                  <linearGradient id="arcGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#c8aa6f" />
                    <stop offset="100%" stopColor="#00f0ff" />
                  </linearGradient>
                </defs>
              </svg>
              
              <div className="absolute flex flex-col items-center mt-4">
                <span className="text-5xl font-black text-white leading-none">85<span className="text-primary text-2xl">%</span></span>
                <span className="text-[10px] text-white/40 font-black uppercase tracking-[0.2em] mt-2">Critical Match</span>
              </div>
            </div>

            {/* CONTEXTUAL FACTORS: Detail-rich list */}
            <div className="space-y-4 relative z-10">
              <div className="flex items-center justify-between mb-2">
                <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest border-b border-white/5 pb-2 flex-1">Neural Detection Insight</p>
                <span className="material-symbols-outlined text-slate-500 text-xs ml-2 cursor-help">info</span>
              </div>
              
              {detectionFactors.map((f, i) => (
                <div key={i} className="flex flex-col gap-1 group/factor cursor-help">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className={`material-symbols-outlined text-sm ${f.color}`}>{f.icon}</span>
                      <span className="text-xs text-slate-300 font-bold group-hover/factor:text-white transition-colors">{f.label}</span>
                    </div>
                    <span className={`text-[10px] font-black uppercase bg-white/5 px-1.5 py-0.5 rounded ${f.color}`}>{f.value}</span>
                  </div>
                  <p className="text-[9px] text-slate-500 font-medium ml-6 leading-tight group-hover/factor:text-slate-400 transition-colors">
                    {f.context}
                  </p>
                </div>
              ))}
            </div>

            <div className="mt-8 p-4 rounded-lg bg-primary/10 border border-primary/20 relative group-hover:bg-primary/20 transition-all">
              <div className="flex gap-3">
                <span className="material-symbols-outlined text-primary text-xl">psychology</span>
                <div>
                  <p className="text-[11px] font-bold text-primary uppercase mb-1">AI Analyst Verdict</p>
                  <p className="text-[10px] text-white/80 leading-relaxed">
                    Account exhibits patterns typical of <b>alternate identity</b> play. Combat effectiveness is in the 99th percentile for current bracket. 
                    <span className="block mt-2 text-primary/60 underline decoration-dotted underline-offset-4 cursor-pointer">Export full behavioral report</span>
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="glass-panel p-6 rounded-xl">
             <div className="flex justify-between items-center mb-4">
              <p className="text-slate-400 text-xs font-bold uppercase tracking-[0.2em]">Progression Analysis</p>
              <span className="text-hextech-blue text-xs font-bold">+15% Gain</span>
            </div>
            <div className="h-40 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={progressionData}>
                  <defs>
                    <linearGradient id="colorGain" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#c8aa6f" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#c8aa6f" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="game" hide />
                  <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(200, 170, 111, 0.2)', strokeWidth: 1 }} />
                  <Area 
                    type="monotone" 
                    dataKey="gain" 
                    stroke="#c8aa6f" 
                    fill="url(#colorGain)" 
                    strokeWidth={3} 
                    activeDot={{ r: 6, fill: '#c8aa6f', stroke: '#0a0e13', strokeWidth: 2 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
              <div className="flex justify-between px-2 text-[10px] font-bold text-slate-500 mt-2">
                <span>START</span>
                <span>RECENT SESSIONS</span>
              </div>
            </div>
          </div>
        </div>

        {/* Match History */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <span className="material-symbols-outlined text-primary">history</span> Match History
            </h2>
            <div className="flex gap-2">
              <span className="px-3 py-1 bg-primary text-background-dark text-[10px] font-bold rounded uppercase cursor-pointer">All</span>
              <span className="px-3 py-1 bg-white/5 text-slate-400 text-[10px] font-bold rounded uppercase cursor-pointer hover:text-primary transition-colors">Ranked</span>
            </div>
          </div>

          {[
            { res: 'Victory', color: 'win-accent', bg: 'bg-win', champ: 'Lee Sin', kda: '12 / 2 / 8', score: '10.00', rank: 'MVP', scoreVal: 9.8, img: 'https://picsum.photos/seed/lee/100' },
            { res: 'Defeat', color: 'red-500', bg: 'bg-loss', champ: 'Ahri', kda: '4 / 7 / 12', score: '2.28', rank: 'ACE', scoreVal: 6.2, img: 'https://picsum.photos/seed/ahri/100' },
            { res: 'Victory', color: 'win-accent', bg: 'bg-win', champ: 'Yasuo', kda: '9 / 1 / 4', score: '13.00', rank: 'S+', scoreVal: 9.9, img: 'https://picsum.photos/seed/yasuo/100' },
          ].map((match, i) => (
            <div key={i} className={`${match.bg} border-l-4 border-${match.color === 'win-accent' ? 'hextech-blue' : 'red-500'} rounded-lg overflow-hidden flex items-center p-4 transition-transform hover:scale-[1.01] cursor-pointer group/item`}>
              <div className="flex flex-col items-center w-24 border-r border-slate-700/50 pr-4">
                <span className={`text-xs font-bold ${match.color === 'win-accent' ? 'text-hextech-blue' : 'text-red-500'} uppercase tracking-wider`}>{match.res}</span>
                <span className="text-[10px] text-slate-400 font-medium">Ranked Solo</span>
                <span className="text-[10px] text-slate-500 mt-2">24:12</span>
              </div>
              <div className="flex items-center gap-4 px-4 flex-1">
                <div className="relative size-14">
                  <img className="size-full rounded-lg border border-primary/40 object-cover" src={match.img} alt={match.champ} />
                  <span className="absolute -bottom-1 -right-1 bg-black border border-primary/40 text-[9px] px-1 rounded text-white font-black">18</span>
                </div>
                <div className="flex flex-col items-center flex-1">
                  <p className="text-xl font-black text-slate-100 tabular-nums group-hover/item:text-primary transition-colors">{match.kda}</p>
                  <p className="text-xs text-slate-400 font-bold">{match.score} <span className="text-primary">KDA</span></p>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <div className="flex gap-1">
                    {[1,2,3,4,5].map(x => <div key={x} className="size-6 bg-slate-800 border border-slate-700 rounded-sm"></div>)}
                  </div>
                  <div className="text-[10px] text-slate-500 font-medium">CS 245 (9.2)</div>
                </div>
              </div>
              <div className="flex flex-col items-center gap-2 px-4 border-l border-slate-700/50">
                <div className="size-10 rounded-full border-2 border-primary flex items-center justify-center bg-background-dark shadow-[0_0_10px_rgba(200,170,111,0.2)]">
                  <span className="text-primary font-black text-xs">{match.rank}</span>
                </div>
                <span className="text-[10px] font-bold text-primary whitespace-nowrap">SCORE: {match.scoreVal}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Strategy Insights Section */}
      <section className="mt-12 pt-8 border-t border-primary/20">
        <header className="flex items-center justify-between mb-8">
          <h2 className="text-white text-xl font-black flex items-center gap-3">
            <span className="material-symbols-outlined text-primary text-3xl">lightbulb</span> 
            AI STRATEGY INSIGHTS
          </h2>
          <span className="text-[10px] text-primary bg-primary/10 border border-primary/20 px-3 py-1 rounded-full font-bold uppercase tracking-widest">v14.2 Optimized</span>
        </header>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[
            { title: 'Win Condition', desc: 'Focus on early dragons. You have an 82% win rate when taking the first 2 drakes.', icon: 'trophy' },
            { title: 'Weakness Detected', desc: 'Vision score drops significantly between 15-20 minutes. Consider buying more Control Wards.', icon: 'visibility_off' },
            { title: 'Strongest Synergy', desc: 'You perform best when paired with aggressive junglers like Nidalee or Jarvan IV.', icon: 'handshake' },
          ].map((insight, i) => (
            <div key={i} className="p-6 rounded-lg bg-white/5 border border-primary/10 hover:border-primary/40 hover:bg-primary/5 transition-all cursor-default group/insight">
              <div className="flex items-center gap-3 mb-3">
                <span className="material-symbols-outlined text-primary group-hover/insight:scale-110 transition-transform">{insight.icon}</span>
                <span className="text-primary font-black text-xs uppercase tracking-widest">{insight.title}</span>
              </div>
              <p className="text-slate-300 text-sm leading-relaxed">{insight.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="mt-12 py-12 border-t border-primary/10 text-center flex flex-col items-center gap-4">
        <div className="flex gap-4 opacity-30 hover:opacity-100 transition-opacity">
           <span className="material-symbols-outlined text-white cursor-pointer hover:text-primary">share</span>
           <span className="material-symbols-outlined text-white cursor-pointer hover:text-primary">download</span>
           <span className="material-symbols-outlined text-white cursor-pointer hover:text-primary">bookmark</span>
        </div>
        <p className="text-slate-500 text-[10px] uppercase font-bold tracking-[0.3em]">Hextech Insights &bull; Neural Analytics System &bull; 2024</p>
        <p className="text-slate-600 text-[9px] max-w-2xl italic">Hextech Insights isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing League of Legends.</p>
      </footer>
    </div>
  );
};

export default Profile;
