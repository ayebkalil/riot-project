
import React from 'react';

interface GaugeProps {
  value: number; // 0 to 100
  size?: number;
  label?: string;
  subLabel?: string;
}

const Gauge: React.FC<GaugeProps> = ({ value, size = 300, label, subLabel }) => {
  const radius = size / 2 - 20;
  const circumference = Math.PI * radius;
  const strokeDashoffset = circumference - (value / 100) * circumference;

  return (
    <div className="relative flex flex-col items-center justify-center overflow-hidden" style={{ width: size, height: size / 2 + 20 }}>
      <svg width={size} height={size} className="transform -rotate-180 origin-center translate-y-[25%]">
        {/* Background Arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#35322c"
          strokeWidth="20"
          strokeDasharray={circumference}
          strokeDashoffset={0}
          strokeLinecap="round"
        />
        {/* Fill Arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#c8aa6f"
          strokeWidth="20"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div className="absolute top-[60%] flex flex-col items-center">
        <span className="text-5xl font-black text-white leading-none">{value}%</span>
        {label && <span className="text-[10px] font-bold text-primary mt-1 uppercase tracking-widest">{label}</span>}
        {subLabel && <span className="text-[8px] text-white/40 mt-1">{subLabel}</span>}
      </div>
    </div>
  );
};

export default Gauge;
