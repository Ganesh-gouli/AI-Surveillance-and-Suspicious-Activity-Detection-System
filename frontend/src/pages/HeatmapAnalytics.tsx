import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  Flame,
  Clock,
  TrendingUp,
  Activity,
  ShieldAlert,
  Calendar,
  Layers
} from 'lucide-react';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { api } from '../services/api';

export const HeatmapAnalytics: React.FC = () => {
  const [timeframe, setTimeframe] = useState<'today' | '7days' | '30days'>('today');
  const [analyticsData, setAnalyticsData] = useState<any>(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      const data = await api.getAnalyticsOverview();
      setAnalyticsData(data);
    };
    fetchAnalytics();
  }, []);

  const hourlyData = analyticsData?.hourly_trends || [
    { hour: "00:00", people: 6, threats: 0 },
    { hour: "04:00", people: 4, threats: 0 },
    { hour: "08:00", people: 48, threats: 2 },
    { hour: "12:00", people: 95, threats: 3 },
    { hour: "14:00", people: 110, threats: 4 },
    { hour: "16:00", people: 88, threats: 2 },
    { hour: "18:00", people: 64, threats: 3 },
    { hour: "22:00", people: 19, threats: 1 }
  ];

  const activityPieData = analyticsData?.activity_distribution || [
    { name: "Sitting", count: 420, color: "#00f0ff" },
    { name: "Running", count: 215, color: "#38bdf8" },
    { name: "Using Laptop", count: 188, color: "#6366f1" },
    { name: "Walking", count: 164, color: "#818cf8" },
    { name: "Eating", count: 92, color: "#34d399" },
    { name: "Drinking", count: 68, color: "#10b981" },
    { name: "Cycling", count: 45, color: "#fbbf24" },
    { name: "Fighting", count: 18, color: "#ef4444" },
    { name: "Sleeping", count: 16, color: "#f87171" }
  ];

  const severityData = analyticsData?.severity_distribution || [
    { severity: "CRITICAL", count: 4, color: "#ef4444" },
    { severity: "HIGH", count: 7, color: "#f59e0b" },
    { severity: "MEDIUM", count: 5, color: "#38bdf8" },
    { severity: "LOW", count: 2, color: "#10b981" }
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header & Timeframe Switch */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#0a0f19] border border-slate-800 p-4 rounded-xl">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <BarChart3 className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white font-mono">
              SPATIAL HEATMAPS & AI BEHAVIOUR ANALYTICS
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              Dwell Density Mapping • 24h Activity Distribution • Threat Severity Breakdown
            </p>
          </div>
        </div>

        {/* Timeframe Filter */}
        <div className="flex items-center space-x-1.5 bg-slate-900 border border-slate-800 p-1 rounded-lg text-xs font-mono">
          <button
            onClick={() => setTimeframe('today')}
            className={`px-3 py-1.5 rounded transition-colors ${
              timeframe === 'today' ? 'bg-cyan-500/20 text-cyan-300 font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            TODAY
          </button>
          <button
            onClick={() => setTimeframe('7days')}
            className={`px-3 py-1.5 rounded transition-colors ${
              timeframe === '7days' ? 'bg-cyan-500/20 text-cyan-300 font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            7 DAYS
          </button>
          <button
            onClick={() => setTimeframe('30days')}
            className={`px-3 py-1.5 rounded transition-colors ${
              timeframe === '30days' ? 'bg-cyan-500/20 text-cyan-300 font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            30 DAYS
          </button>
        </div>
      </div>

      {/* 5 KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <div className="soc-panel p-4 rounded-xl">
          <span className="text-slate-400 text-xs font-mono block">MOST DETECTED</span>
          <span className="text-lg font-bold text-cyan-400 font-mono mt-1 block">Sitting / Walking</span>
          <span className="text-[10px] text-slate-500 font-mono">47.6% of all detections</span>
        </div>

        <div className="soc-panel p-4 rounded-xl">
          <span className="text-slate-400 text-xs font-mono block">MOST ACTIVE CAMERA</span>
          <span className="text-lg font-bold text-white font-mono mt-1 block">CAM-004 Parking</span>
          <span className="text-[10px] text-slate-500 font-mono">112 people / hr</span>
        </div>

        <div className="soc-panel p-4 rounded-xl">
          <span className="text-slate-400 text-xs font-mono block">PEAK THREAT HOUR</span>
          <span className="text-lg font-bold text-amber-400 font-mono mt-1 block">14:00 ─ 16:00</span>
          <span className="text-[10px] text-slate-500 font-mono">4 incidents logged</span>
        </div>

        <div className="soc-panel p-4 rounded-xl">
          <span className="text-slate-400 text-xs font-mono block">SUSPICIOUS EVENTS</span>
          <span className="text-lg font-bold text-red-400 font-mono mt-1 block">18 Incidents</span>
          <span className="text-[10px] text-red-400/80 font-mono">4 critical open</span>
        </div>

        <div className="soc-panel p-4 rounded-xl">
          <span className="text-slate-400 text-xs font-mono block">AVG AI CONFIDENCE</span>
          <span className="text-lg font-bold text-emerald-400 font-mono mt-1 block">94.6%</span>
          <span className="text-[10px] text-emerald-400/80 font-mono">High model certainty</span>
        </div>
      </div>

      {/* Main Grid: Spatial Heatmap Overlay (Left) & Hourly Activity Chart (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 2D Spatial Heatmap Viewport */}
        <div className="soc-panel p-4 rounded-xl space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-xs font-mono font-bold text-slate-200">
              <Flame className="w-4 h-4 text-amber-400" />
              <span>SPATIAL MOVEMENT & DWELL HEATMAP</span>
            </div>
            <span className="text-[10px] font-mono text-cyan-400">CAM-001 MAIN ATRIUM</span>
          </div>

          <div className="relative aspect-video w-full bg-slate-950 rounded-xl border border-slate-800 overflow-hidden surveillance-feed">
            <div className="absolute inset-0 bg-gradient-to-br from-slate-950 to-[#070d18] opacity-80" />

            {/* Heatmap Gaussian Spots */}
            <div
              className="absolute w-44 h-44 rounded-full pointer-events-none blur-2xl opacity-60"
              style={{
                left: '20%',
                top: '30%',
                background: 'radial-gradient(circle, #ef4444 0%, #f59e0b 40%, transparent 70%)'
              }}
            />
            <div
              className="absolute w-56 h-56 rounded-full pointer-events-none blur-2xl opacity-70"
              style={{
                left: '50%',
                top: '35%',
                background: 'radial-gradient(circle, #ef4444 0%, #00f0ff 40%, transparent 70%)'
              }}
            />
            <div
              className="absolute w-36 h-36 rounded-full pointer-events-none blur-2xl opacity-50"
              style={{
                left: '75%',
                top: '55%',
                background: 'radial-gradient(circle, #f59e0b 0%, #10b981 40%, transparent 70%)'
              }}
            />

            {/* Scale legend */}
            <div className="absolute bottom-3 left-3 bg-black/80 border border-slate-800 px-3 py-1.5 rounded text-[10px] font-mono flex items-center space-x-3 text-slate-300">
              <span>Low Density</span>
              <div className="w-20 h-2 rounded bg-gradient-to-r from-emerald-400 via-amber-400 to-red-500" />
              <span>High Dwell</span>
            </div>
          </div>
        </div>

        {/* Hourly Trend Chart */}
        <div className="soc-panel p-4 rounded-xl space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-xs font-mono font-bold text-slate-200">
              <Clock className="w-4 h-4 text-cyan-400" />
              <span>HOURLY OCCUPANCY & SUSPICIOUS INCIDENTS</span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">24H TIMELINE</span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={hourlyData}>
                <defs>
                  <linearGradient id="colorPeople" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00f0ff" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#00f0ff" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorThreats" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="hour" stroke="#64748b" fontSize={10} fontStyle="JetBrains Mono" />
                <YAxis stroke="#64748b" fontSize={10} fontStyle="JetBrains Mono" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0b1019', borderColor: '#1e293b', borderRadius: '8px', fontSize: '11px', fontFamily: 'JetBrains Mono' }}
                />
                <Area type="monotone" dataKey="people" stroke="#00f0ff" fillOpacity={1} fill="url(#colorPeople)" name="People Detections" />
                <Area type="monotone" dataKey="threats" stroke="#ef4444" fillOpacity={1} fill="url(#colorThreats)" name="Threat Events" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Bottom Grid: 8-Class Activity Distribution & Threat Severity Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 8-Class Distribution */}
        <div className="soc-panel p-4 rounded-xl space-y-3">
          <div className="text-xs font-mono font-bold text-slate-200">
            8-CLASS HUMAN ACTIVITY CLASSIFICATION SPREAD
          </div>
          <div className="h-60 w-full flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={activityPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={4}
                  dataKey="count"
                  nameKey="name"
                >
                  {activityPieData.map((entry: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#0b1019', borderColor: '#1e293b', borderRadius: '8px', fontSize: '11px', fontFamily: 'JetBrains Mono' }}
                />
                <Legend formatter={(value) => <span className="text-xs font-mono text-slate-300 capitalize">{value}</span>} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Severity Bar Chart */}
        <div className="soc-panel p-4 rounded-xl space-y-3">
          <div className="text-xs font-mono font-bold text-slate-200">
            ALERT SEVERITY INCIDENT BREAKDOWN
          </div>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={severityData}>
                <XAxis dataKey="severity" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0b1019', borderColor: '#1e293b', borderRadius: '8px', fontSize: '11px', fontFamily: 'JetBrains Mono' }}
                />
                <Bar dataKey="count" fill="#38bdf8" radius={[4, 4, 0, 0]}>
                  {severityData.map((entry: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
