import React, { useState, useEffect } from 'react';
import {
  AlertTriangle,
  Search,
  Filter,
  Eye,
  CheckCircle,
  FileDown,
  Clock,
  ShieldAlert,
  ArrowUpDown
} from 'lucide-react';
import { IncidentDetailModal } from '../components/incidents/IncidentDetailModal';
import { api } from '../services/api';
import { Incident, SeverityLevel, IncidentStatus } from '../types';

export const Incidents: React.FC = () => {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [severityFilter, setSeverityFilter] = useState('ALL');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [cameraFilter, setCameraFilter] = useState('ALL');

  const loadIncidents = async () => {
    const data = await api.getIncidents({
      severity: severityFilter,
      status: statusFilter,
      camera_id: cameraFilter,
      search: searchQuery
    });
    setIncidents(data);
  };

  useEffect(() => {
    loadIncidents();
  }, [severityFilter, statusFilter, cameraFilter, searchQuery]);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#0a0f19] border border-slate-800 p-4 rounded-xl">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white font-mono">
              SECURITY INCIDENTS & THREAT AUDIT LOG
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              SOC Evidence Repository • Temporal Anomaly Investigations • Resolution Workflow
            </p>
          </div>
        </div>

        <button
          onClick={() => alert("Exporting all incidents as CSV / JSON Docket...")}
          className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono font-bold flex items-center space-x-1.5 border border-slate-700"
        >
          <FileDown className="w-4 h-4 text-cyan-400" />
          <span>EXPORT INCIDENT LOG</span>
        </button>
      </div>

      {/* Filter Bar */}
      <div className="bg-[#0b1018] border border-slate-800 p-4 rounded-xl flex flex-wrap items-center gap-3">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px]">
          <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by ID, activity, location, notes..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-9 pr-3 py-1.5 text-xs font-mono text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400"
          />
        </div>

        {/* Severity Filter */}
        <div className="flex items-center space-x-1.5 text-xs font-mono">
          <span className="text-slate-400">SEVERITY:</span>
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white focus:outline-none focus:border-cyan-400"
          >
            <option value="ALL">ALL SEVERITIES</option>
            <option value="CRITICAL">CRITICAL</option>
            <option value="HIGH">HIGH</option>
            <option value="MEDIUM">MEDIUM</option>
            <option value="LOW">LOW</option>
          </select>
        </div>

        {/* Status Filter */}
        <div className="flex items-center space-x-1.5 text-xs font-mono">
          <span className="text-slate-400">STATUS:</span>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white focus:outline-none focus:border-cyan-400"
          >
            <option value="ALL">ALL STATUSES</option>
            <option value="OPEN">OPEN</option>
            <option value="INVESTIGATING">INVESTIGATING</option>
            <option value="RESOLVED">RESOLVED</option>
          </select>
        </div>
      </div>

      {/* Incidents Table */}
      <div className="soc-panel rounded-xl overflow-hidden border border-slate-800 shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#080d16] text-slate-400 border-b border-slate-800 select-none">
              <tr>
                <th className="py-3.5 px-4 font-bold">INCIDENT ID</th>
                <th className="py-3.5 px-4 font-bold">TIMESTAMP</th>
                <th className="py-3.5 px-4 font-bold">CAMERA / LOCATION</th>
                <th className="py-3.5 px-4 font-bold">ACTIVITY TYPE</th>
                <th className="py-3.5 px-4 font-bold">SEVERITY</th>
                <th className="py-3.5 px-4 font-bold">CONFIDENCE</th>
                <th className="py-3.5 px-4 font-bold">STATUS</th>
                <th className="py-3.5 px-4 font-bold text-right">ACTIONS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {incidents.map((inc) => {
                const isCrit = inc.severity === 'CRITICAL';
                const isHigh = inc.severity === 'HIGH';
                const isResolved = inc.status === 'RESOLVED';

                return (
                  <tr
                    key={inc.id}
                    className="hover:bg-slate-800/40 transition-colors group cursor-pointer"
                    onClick={() => setSelectedIncident(inc)}
                  >
                    <td className="py-3.5 px-4 font-bold text-white group-hover:text-cyan-400">
                      {inc.id}
                    </td>
                    <td className="py-3.5 px-4 text-slate-300">
                      {new Date(inc.timestamp).toLocaleString()}
                    </td>
                    <td className="py-3.5 px-4">
                      <span className="font-bold text-cyan-300">{inc.camera_id}</span>
                      <span className="text-slate-400 block text-[11px]">{inc.location}</span>
                    </td>
                    <td className="py-3.5 px-4 font-bold text-slate-100">
                      {inc.activity}
                    </td>
                    <td className="py-3.5 px-4">
                      <span className={`px-2 py-0.5 rounded font-bold text-[10px] ${
                        isCrit
                          ? 'bg-red-500/20 text-red-400 border border-red-500/40'
                          : isHigh
                          ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                          : 'bg-blue-500/20 text-blue-400 border border-blue-500/40'
                      }`}>
                        {inc.severity}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-emerald-400 font-bold">
                      {inc.confidence}%
                    </td>
                    <td className="py-3.5 px-4">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        isResolved
                          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                          : 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                      }`}>
                        {inc.status}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <button
                        onClick={(e) => { e.stopPropagation(); setSelectedIncident(inc); }}
                        className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono font-bold"
                      >
                        VIEW
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Incident Detail Modal */}
      <IncidentDetailModal
        incident={selectedIncident}
        onClose={() => setSelectedIncident(null)}
        onResolve={async (id, notes) => {
          await api.updateIncident(id, { status: 'RESOLVED', resolution_notes: notes });
          loadIncidents();
          setSelectedIncident(null);
        }}
      />
    </div>
  );
};
