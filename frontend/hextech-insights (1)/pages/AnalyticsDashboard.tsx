
import React from 'react';
import { BarChart, Bar, ResponsiveContainer, XAxis, Tooltip, Cell } from 'recharts';

const rankData = [
  { rank: 'Ir', count: 15, color: '#475569' },
  { rank: 'Br', count: 25, color: '#475569' },
  { rank: 'Si', count: 35, color: '#475569' },
  { rank: 'Go', count: 60, color: '#475569' },
  { rank: 'Pl', count: 95, color: '#c8aa6f' },
  { rank: 'Em', count: 85, color: '#c8aa6f' },
  { rank: 'Di', count: 55, color: '#475569' },
  { rank: 'Ma', count: 30, color: '#475569' },
  { rank: 'Gm', count: 15, color: '#475569' },
  { rank: 'Ch', count: 8, color: '#00bcda' },
];

const AnalyticsDashboard: React.FC = () => {
  return (
    <div className="p-8 space-y-8 max-w-[1600px] mx-auto">
      {/* Hero Stats Row */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          { label: 'Players Tracked', value: '12,548,201', trend: '+1.2% this week', trendUp: true, icon: 'group' },
          { label: 'Global Win Rate', value: '50.24%', trend: '-0.05% vs Patch 14.1', trendUp: false, icon: 'insights' },
          { label: 'Top Champion', value: "Kai'Sa", sub: '52.1% Win Rate', icon: 'trophy', image: 'https://picsum.photos/seed/kaisa/64' },
          { label: 'Smurf Alerts', value: '1,402', trend: 'Critical anomalies detected', trendUp: true, icon: 'radar', isBlue: true },
        ].map((stat, i) => (
          <div key={i} className={`relative p-6 rounded-lg border border-primary/30 bg-[#1e1a14] overflow-hidden`}>
            <div className="flex justify-between items-start mb-2 relative z-10">
              <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">{stat.label}</p>
              <span className={`material-symbols-outlined ${stat.isBlue ? 'text-hextech-blue' : 'text-primary'} text-xl`}>{stat.icon}</span>
            </div>
            {stat.image ? (
              <div className="flex items-center gap-3">
                <div className="size-9 rounded bg-slate-800 border border-primary/40 bg-cover bg-center" style={{ backgroundImage: `url(${stat.image})` }}></div>
                <div>
                  <p className="text-xl font-extrabold text-white">{stat.value}</p>
                  <p className="text-xs text-slate-500">{stat.sub}</p>
                </div>
              </div>
            ) : (
              <p className="text-3xl font-extrabold text-white">{stat.value}</p>
            )}
            {stat.trend && (
              <div className="flex items-center gap-1 mt-2 relative z-10">
                {!stat.isBlue && <span className={`material-symbols-outlined text-xs ${stat.trendUp ? 'text-green-500' : 'text-red-500'}`}>{stat.trendUp ? 'trending_up' : 'trending_down'}</span>}
                <p className={`text-xs font-bold ${stat.isBlue ? 'text-hextech-blue' : (stat.trendUp ? 'text-green-500' : 'text-red-500')}`}>{stat.trend}</p>
              </div>
            )}
            {stat.isBlue && <div className="absolute -bottom-4 -right-4 size-24 bg-hextech-blue/10 blur-3xl rounded-full"></div>}
          </div>
        ))}
      </section>

      {/* Main Charts Grid */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 glass-panel rounded-lg p-6">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <span className="material-symbols-outlined text-primary">bar_chart</span>
                Global Rank Distribution
              </h2>
              <p className="text-xs text-slate-500">Data reflects Solo/Duo queue across all regions (Current Patch)</p>
            </div>
            <div className="flex gap-2">
              <button className="px-3 py-1 bg-primary/20 text-primary text-[10px] font-bold rounded border border-primary/30 uppercase">Area View</button>
              <button className="px-3 py-1 bg-white/5 text-slate-400 text-[10px] font-bold rounded border border-white/5 uppercase">Bar View</button>
            </div>
          </div>
          
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rankData}>
                <XAxis dataKey="rank" axisLine={false} tickLine={false} tick={{ fill: '#64748b', fontSize: 10, fontWeight: 'bold' }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0a1428', border: '1px solid rgba(200, 170, 111, 0.3)', borderRadius: '8px' }}
                  itemStyle={{ color: '#c8aa6f' }}
                  cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {rankData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass-panel rounded-lg p-6 flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <span className="material-symbols-outlined text-hextech-blue">shutter_speed</span>
              Smurf Watch
            </h2>
            <span className="text-[10px] font-bold text-hextech-blue bg-hextech-blue/10 px-2 py-1 rounded">LIVE FEED</span>
          </div>
          
          <div className="space-y-4 flex-1">
            {[
              { name: 'Hide on Bush v2', wr: '92%', rank: 'Em IV → Di II', status: 'Critical', color: 'text-primary' },
              { name: 'I Only Play Jinx', wr: '88%', rank: 'Unranked → Pl I', status: 'High Conf.', color: 'text-hextech-blue' },
              { name: 'L9 Fanboy 2024', wr: '95%', rank: 'Go II → Em I', status: 'Critical', color: 'text-primary' },
            ].map((smurf, i) => (
              <div key={i} className="p-3 rounded border border-white/5 bg-white/5 hover:border-hextech-blue/30 transition-all cursor-pointer group">
                <div className="flex justify-between items-start mb-1">
                  <p className="text-sm font-bold text-white group-hover:text-hextech-blue transition-colors">{smurf.name}</p>
                  <p className="text-[10px] font-bold text-green-500">{smurf.wr} WR</p>
                </div>
                <div className="flex items-center justify-between">
                  <p className="text-[10px] text-slate-500">{smurf.rank}</p>
                  <div className="flex gap-1 items-center">
                    <span className={`size-1.5 rounded-full ${smurf.color === 'text-primary' ? 'bg-primary animate-pulse' : 'bg-hextech-blue'}`}></span>
                    <p className={`text-[9px] font-bold ${smurf.color} uppercase`}>{smurf.status}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          <button className="w-full mt-6 py-3 border border-primary/40 text-primary text-xs font-bold rounded-lg hover:bg-primary hover:text-background-dark transition-all">
            VIEW ALL DETECTIONS
          </button>
        </div>
      </section>

      {/* Regional Activity Section */}
      <section className="grid grid-cols-1 xl:grid-cols-2 gap-8 pb-8">
        <div className="glass-panel rounded-lg p-6">
          <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-6">
            <span className="material-symbols-outlined text-primary">language</span>
            Regional Activity
          </h2>
          <div className="relative w-full h-48 bg-slate-900 rounded overflow-hidden">
            <div 
              className="absolute inset-0 bg-cover bg-center opacity-40 grayscale" 
              style={{ backgroundImage: `url('https://picsum.photos/seed/map/800/400')` }}
            ></div>
            <div className="absolute top-1/4 left-1/4 size-3 bg-primary rounded-full blur-[2px] shadow-[0_0_10px_#c8aa6f]"></div>
            <div className="absolute top-1/3 left-1/2 size-4 bg-hextech-blue rounded-full blur-[2px] shadow-[0_0_10px_#00bcda]"></div>
            <div className="absolute top-1/2 right-1/4 size-6 bg-primary rounded-full blur-[2px] shadow-[0_0_15px_#c8aa6f]"></div>
            <div className="absolute bottom-4 left-4 bg-background-dark/80 backdrop-blur-sm p-3 rounded border border-primary/30">
              <p className="text-[10px] text-slate-400 font-bold uppercase mb-1">Highest Traffic</p>
              <p className="text-sm font-extrabold text-white">KR: Seoul Server</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 flex flex-col justify-center">
            <p className="text-[10px] text-primary font-bold uppercase tracking-wider mb-1">Queue Health</p>
            <p className="text-2xl font-black text-white">OPTIMAL</p>
            <p className="text-[10px] text-slate-500 mt-1">Avg Queue: 1m 24s</p>
          </div>
          <div className="bg-hextech-blue/5 border border-hextech-blue/20 rounded-lg p-4 flex flex-col justify-center">
            <p className="text-[10px] text-hextech-blue font-bold uppercase tracking-wider mb-1">API Latency</p>
            <p className="text-2xl font-black text-white">24ms</p>
            <p className="text-[10px] text-slate-500 mt-1">Riot Game Data v4</p>
          </div>
          <div className="sm:col-span-2 glass-panel rounded-lg p-4 flex items-center gap-4">
            <div className="size-12 rounded bg-primary/20 flex items-center justify-center border border-primary/40">
              <span className="material-symbols-outlined text-primary">security</span>
            </div>
            <div>
              <p className="text-sm font-bold text-white">Vanguard Security Status</p>
              <p className="text-xs text-slate-500">Last scan completed 4 minutes ago. 0 major bypasses detected.</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default AnalyticsDashboard;
