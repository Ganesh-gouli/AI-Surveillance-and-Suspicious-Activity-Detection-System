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
  ArrowUpDown,
  FileSpreadsheet,
  FileCode,
  RotateCcw,
  ShieldCheck,
  Activity
} from 'lucide-react';
import { IncidentDetailModal } from '../components/incidents/IncidentDetailModal';
import { useToast } from '../context/ToastContext';
import { api } from '../services/api';
import { Incident, SeverityLevel, IncidentStatus } from '../types';

export const Incidents: React.FC = () => {
  const { success, info } = useToast();
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [severityFilter, setSeverityFilter] = useState('ALL');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [cameraFilter, setCameraFilter] = useState('ALL');
  const [isLoading, setIsLoading] = useState(false);

  const loadIncidents = async () => {
    setIsLoading(true);
    try {
      const data = await api.getIncidents({
        severity: severityFilter !== 'ALL' ? severityFilter : undefined,
        status: statusFilter !== 'ALL' ? statusFilter : undefined,
        camera_id: cameraFilter !== 'ALL' ? cameraFilter : undefined,
        search: searchQuery.trim() || undefined
      });
      setIncidents(data);
    } catch (err) {
      console.error('Error loading incidents:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadIncidents();
  }, [severityFilter, statusFilter, cameraFilter, searchQuery]);

  // Real client-side CSV Export
  const handleExportCSV = () => {
    if (incidents.length === 0) {
      info('No Incidents', 'There are no incidents currently matching the active filters to export.');
      return;
    }

    const headers = ['Incident ID', 'Timestamp', 'Camera ID', 'Location', 'Activity', 'Severity', 'Confidence', 'Status', 'Details', 'Resolution Notes'];
    const rows = incidents.map((inc) => [
      `"${inc.id}"`,
      `"${inc.timestamp}"`,
      `"${inc.camera_id}"`,
      `"${inc.location}"`,
      `"${inc.activity}"`,
      `"${inc.severity}"`,
      `"${inc.confidence}%"`,
      `"${inc.status}"`,
      `"${(inc.details || '').replace(/"/g, '""')}"`,
      `"${(inc.resolution_notes || '').replace(/"/g, '""')}"`
    ]);

    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `sentinelvision_incidents_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    success('CSV Exported', `Exported ${incidents.length} incident records.`);
  };

  // Real client-side JSON Export
  const handleExportJSON = () => {
    if (incidents.length === 0) {
      info('No Incidents', 'No incident records to export.');
      return;
    }

    const jsonString = JSON.stringify(incidents, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `sentinelvision_incidents_${new Date().toISOString().slice(0, 10)}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    success('JSON Exported', `Exported ${incidents.length} incident records in structured JSON.`);
  };

  const handleResetFilters = () => {
    setSearchQuery('');
    setSeverityFilter('ALL');
    setStatusFilter('ALL');
    setCameraFilter('ALL');
  };

  const totalIncidentsCount = incidents.length;
  const criticalCount = incidents.filter((i) => i.severity === 'CRITICAL').length;
  const openCount = incidents.filter((i) => i.status === 'OPEN').length;
  const resolvedCount = incidents.filter((i) => i.status === 'RESOLVED').length;

  return (
    <div className="p-4 sm:p-6 space-y-6 cyber-grid-bg min-h-full">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#0a0f19] border border-slate-800 p-4 sm:p-5 rounded-2xl shadow-xl">
        <div className="flex items-center space-x-3.5">
          <div className="p-2.5 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400">
            <ShieldAlert className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h1 className="text-base sm:text-lg font-bold text-white font-mono tracking-tight">
              SECURITY INCIDENTS & THREAT AUDIT LOG
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              Temporal Evidence Repository • Verified Incident History • Formal Resolution Pipeline
            </p>
          </div>
        </div>

        {/* Real Export Buttons */}
        <div className="flex items-center space-x-2">
          <button
            onClick={handleExportCSV}
            className="px-3.5 py-2 rounded-xl btn-tactical-secondary text-xs font-mono font-bold flex items-center space-x-1.5 transition-all"
            title="Download CSV report of incidents"
          >
            <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />
            <span>EXPORT CSV</span>
          </button>
          <button
            onClick={handleExportJSON}
            className="px-3.5 py-2 rounded-xl btn-tactical-secondary text-xs font-mono font-bold flex items-center space-x-1.5 transition-all"
            title="Download JSON docket of incidents"
          >
            <FileCode className="w-3.5 h-3.5 text-cyan-400" />
            <span>EXPORT JSON</span>
          </button>
        </div>
      </div>

      {/* Real Statistics Quick Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
        <div className="soc-card p-3.5 rounded-xl space-y-1">
          <span className="text-[10px] font-mono font-bold text-slate-400 tracking-wider">TOTAL FILTERED</span>
          <div className="text-xl font-bold font-mono text-white">{totalIncidentsCount}</div>
        </div>

        <div className="soc-card p-3.5 rounded-xl space-y-1">
          <span className="text-[10px] font-mono font-bold text-red-400 tracking-wider flex items-center space-x-1">
            {criticalCount > 0 && <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-ping" />}
            <span>CRITICAL SEVERITY</span>
          </span>
          <div className="text-xl font-bold font-mono text-red-400">{criticalCount}</div>
        </div>

        <div className="soc-card p-3.5 rounded-xl space-y-1">
          <span className="text-[10px] font-mono font-bold text-amber-400 tracking-wider">OPEN / ACTIVE</span>
          <div className="text-xl font-bold font-mono text-amber-400">{openCount}</div>
        </div>

        <div className="soc-card p-3.5 rounded-xl space-y-1">
          <span className="text-[10px] font-mono font-bold text-emerald-400 tracking-wider">RESOLVED</span>
          <div className="text-xl font-bold font-mono text-emerald-400">{resolvedCount}</div>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="bg-[#0b1018] border border-slate-800/90 p-4 rounded-xl flex flex-wrap items-center gap-3">
        {/* Search */}
        <div className="relative flex-1 min-w-[220px]">
          <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by ID, activity, location, notes..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700/80 rounded-xl pl-9 pr-3 py-2 text-xs font-mono text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400"
          />
        </div>

        {/* Severity Filter */}
        <div className="flex items-center space-x-1.5 text-xs font-mono">
          <span className="text-slate-400">SEVERITY:</span>
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="bg-slate-900 border border-slate-700/80 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-cyan-400 cursor-pointer"
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
            className="bg-slate-900 border border-slate-700/80 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-cyan-400 cursor-pointer"
          >
            <option value="ALL">ALL STATUSES</option>
            <option value="OPEN">OPEN</option>
            <option value="INVESTIGATING">INVESTIGATING</option>
            <option value="RESOLVED">RESOLVED</option>
          </select>
        </div>

        {(searchQuery || severityFilter !== 'ALL' || statusFilter !== 'ALL') && (
          <button
            onClick={handleResetFilters}
            className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono flex items-center space-x-1"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>RESET</span>
          </button>
        )}
      </div>

      {/* Incidents Table / Content */}
      <div className="soc-panel rounded-2xl overflow-hidden border border-slate-800 shadow-2xl">
        {incidents.length === 0 ? (
          <div className="flex flex-col items-center justify-center text-center py-16 px-4 space-y-3">
            <div className="p-4 rounded-full bg-slate-900 border border-slate-800 text-slate-500">
              <ShieldCheck className="w-10 h-10 text-emerald-400" />
            </div>
            <h3 className="text-sm font-mono font-bold text-white">NO MATCHING INCIDENTS</h3>
            <p className="text-xs font-mono text-slate-400 max-w-sm">
              {searchQuery || severityFilter !== 'ALL' || statusFilter !== 'ALL'
                ? 'No incidents matched your active search query and filter criteria.'
                : 'No incidents are currently logged in the system. The surveillance perimeter is operating normally.'}
            </p>
            {(searchQuery || severityFilter !== 'ALL' || statusFilter !== 'ALL') && (
              <button
                onClick={handleResetFilters}
                className="mt-2 px-4 py-2 rounded-xl btn-tactical-primary text-xs font-mono font-bold inline-flex items-center space-x-1.5"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>CLEAR FILTERS</span>
              </button>
            )}
          </div>
        ) : (
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
                  <th className="py-3.5 px-4 font-bold text-right">ACTION</th>
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
                      className="hover:bg-slate-800/50 transition-colors group cursor-pointer"
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
                        <span
                          className={`px-2 py-0.5 rounded font-bold text-[10px] ${
                            isCrit
                              ? 'bg-red-500/20 text-red-400 border border-red-500/40'
                              : isHigh
                              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                              : 'bg-blue-500/20 text-blue-400 border border-blue-500/40'
                          }`}
                        >
                          {inc.severity}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 text-emerald-400 font-bold">
                        {inc.confidence}%
                      </td>
                      <td className="py-3.5 px-4">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            isResolved
                              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                              : 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                          }`}
                        >
                          {inc.status}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedIncident(inc);
                          }}
                          className="px-3 py-1 rounded-lg bg-slate-800 hover:bg-cyan-500/20 hover:text-cyan-300 hover:border-cyan-500/40 border border-slate-700 text-slate-200 text-xs font-mono font-bold transition-all"
                        >
                          INSPECT
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Incident Detail Modal */}
      <IncidentDetailModal
        incident={selectedIncident}
        onClose={() => setSelectedIncident(null)}
        onResolve={async (id, notes) => {
          await api.updateIncident(id, { status: 'RESOLVED', resolution_notes: notes });
          success('Incident Resolved', `Incident ${id} updated to RESOLVED.`);
          loadIncidents();
          setSelectedIncident(null);
        }}
      />
    </div>
  );
};
