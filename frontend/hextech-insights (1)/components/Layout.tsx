
import React from 'react';
import { Link, useLocation } from 'react-router-dom';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();

  const navItems = [
    { label: 'Dashboard', path: '/dashboard', icon: 'dashboard' },
    { label: 'Live Analytics', path: '/profile', icon: 'query_stats' },
    { label: 'Model Metrics', path: '/models', icon: 'analytics' },
    { label: 'Predictions', path: '/predictions', icon: 'radar' },
  ];

  return (
    <div className="flex h-screen overflow-hidden bg-background-dark">
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 bg-[#0a1428] border-r border-primary/20 flex flex-col hidden lg:flex">
        <div className="p-6 flex items-center gap-3">
          <div className="size-10 bg-primary/20 rounded-lg flex items-center justify-center border border-primary/50">
            <span className="material-symbols-outlined text-primary text-2xl">diamond</span>
          </div>
          <div>
            <h1 className="text-white font-bold leading-none tracking-tight">HEX ANALYTICS</h1>
            <p className="text-xs text-primary/70 uppercase tracking-widest mt-1">Version 2.4.0</p>
          </div>
        </div>
        
        <nav className="flex-1 px-4 py-4 space-y-1">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all group cursor-pointer ${
                location.pathname === item.path
                  ? 'bg-primary/10 border-l-2 border-primary text-primary'
                  : 'text-slate-400 hover:bg-white/5'
              }`}
            >
              <span className={`material-symbols-outlined ${location.pathname === item.path ? 'text-primary' : 'group-hover:text-hextech-blue'}`}>
                {item.icon}
              </span>
              <span className={`text-sm font-semibold`}>{item.label}</span>
            </Link>
          ))}
          
          <div className="pt-6 pb-2 px-4">
            <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">System Status</p>
          </div>
          <div className="flex items-center gap-3 px-4 py-3 text-slate-400">
            <span className="material-symbols-outlined text-green-500">database</span>
            <span className="text-sm font-medium">API: Online</span>
          </div>
        </nav>

        <div className="p-4 border-t border-primary/10">
          <div className="flex items-center gap-3 p-2">
            <div 
              className="size-8 rounded-full bg-slate-800 border border-primary/30 bg-cover bg-center" 
              style={{ backgroundImage: `url('https://picsum.photos/seed/user/100')` }}
            ></div>
            <div className="flex-1 overflow-hidden">
              <p className="text-xs font-bold text-white truncate">HextechAdmin</p>
              <p className="text-[10px] text-slate-500 truncate">EUW Region</p>
            </div>
            <span className="material-symbols-outlined text-slate-500 text-sm">settings</span>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <header className="h-16 border-b border-primary/20 flex items-center justify-between px-8 bg-background-dark/80 backdrop-blur-md z-10">
          <div className="flex items-center flex-1 max-w-2xl gap-4">
            <div className="flex-1 relative group">
              <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-primary transition-colors">search</span>
              <input 
                className="w-full bg-slate-900/50 border border-slate-700 rounded-lg py-2 pl-10 pr-4 text-sm focus:ring-1 focus:ring-primary focus:border-primary transition-all text-white placeholder-slate-500" 
                placeholder="Search Summoner Name..." 
                type="text"
              />
            </div>
            <div className="flex items-center bg-slate-900/50 border border-slate-700 rounded-lg px-3 py-2 text-xs font-bold text-slate-300 cursor-pointer hover:border-primary/50 transition-colors">
              NA1 <span className="material-symbols-outlined text-sm ml-2">expand_more</span>
            </div>
          </div>
          <div className="flex items-center gap-6 ml-8">
            <div className="flex gap-4">
              <Link to="/dashboard" className="text-xs font-bold text-primary hover:text-white transition-colors">Global</Link>
              <Link to="/profile" className="text-xs font-bold text-slate-400 hover:text-white transition-colors">Pro Scene</Link>
            </div>
            <div className="h-6 w-px bg-slate-800"></div>
            <button className="relative">
              <span className="material-symbols-outlined text-slate-400 hover:text-primary transition-colors">notifications</span>
              <span className="absolute top-0 right-0 size-2 bg-hextech-blue rounded-full border border-background-dark"></span>
            </button>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto custom-scrollbar">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
