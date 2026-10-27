import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Activity, Cpu, Database, Microscope, CheckCircle, XCircle, Clock, AlertTriangle } from 'lucide-react';
import apiService from '../services/api';

const DISCLAIMER = 'This application is intended for research and educational purposes. It is not a substitute for professional medical diagnosis.';

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const [s, h, m, hlt] = await Promise.allSettled([
          apiService.getStats(),
          apiService.getHistory({ limit: 5 }),
          apiService.getModelInfo(),
          apiService.getHealth(),
        ]);
        if (!mounted) return;
        if (s.status === 'fulfilled') setStats(s.value);
        if (h.status === 'fulfilled') setHistory(h.value);
        if (m.status === 'fulfilled') setModelInfo(m.value);
        if (hlt.status === 'fulfilled') setHealth(hlt.value);
      } catch (e) {
        if (mounted) setError(e.message);
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => { mounted = false; };
  }, []);

  const byClass = stats?.by_class || {};

  return (
    <div className="page-container animate-fade-in">
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">Analysis overview and system status</p>
      </div>

      {/* API status banner */}
      {error && (
        <div className="alert alert-warning" style={{ marginBottom: 24 }}>
          <AlertTriangle size={16} />
          <span>Cannot connect to API server. Make sure the backend is running: <code>uvicorn backend.main:app --reload</code></span>
        </div>
      )}

      {/* Stats cards */}
      <div className="grid-4" style={{ marginBottom: 32 }}>
        <StatCard icon={<Database size={22} />} label="Total Analyses" value={loading ? '—' : (stats?.total_analyses ?? 0)} color="#3b82f6" />
        <StatCard icon={<Activity size={22} />} label="Cyst Detected" value={loading ? '—' : (byClass['Cyst'] ?? 0)} color="#f59e0b" />
        <StatCard icon={<CheckCircle size={22} />} label="Normal" value={loading ? '—' : (byClass['Normal'] ?? 0)} color="#10b981" />
        <StatCard icon={<AlertTriangle size={22} />} label="Tumor Detected" value={loading ? '—' : (byClass['Tumor'] ?? 0)} color="#ef4444" />
      </div>

      <div className="grid-2" style={{ marginBottom: 32 }}>
        {/* Model availability */}
        <div className="card">
          <div className="card-title">Model Availability</div>
          {loading ? (
            <div>
              {[1,2,3].map(i => <div key={i} className="skeleton skeleton-text" style={{ marginBottom: 12 }} />)}
            </div>
          ) : modelInfo ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {modelInfo.models?.map((m) => (
                <div key={m.name} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 0', borderBottom: '1px solid var(--border)' }}>
                  <Cpu size={18} color={m.status === 'loaded' ? '#10b981' : '#ef4444'} />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)' }}>{m.name}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{m.framework}</div>
                  </div>
                  <span className={`badge ${m.status === 'loaded' ? 'badge-normal' : 'badge-red'}`}>
                    {m.status === 'loaded' ? '✓ Loaded' : '✗ Not Trained'}
                  </span>
                  {m.metrics_available && m.accuracy != null && (
                    <span className="badge badge-blue">{(m.accuracy * 100).toFixed(1)}%</span>
                  )}
                </div>
              ))}
              {!modelInfo.any_available && (
                <div className="alert alert-warning" style={{ marginTop: 8 }}>
                  <AlertTriangle size={14} />
                  <span>No models loaded. Run the ML training scripts first.</span>
                </div>
              )}
            </div>
          ) : (
            <div className="empty-state" style={{ padding: '24px 0' }}>
              <p>Model info unavailable</p>
            </div>
          )}
        </div>

        {/* System status */}
        <div className="card">
          <div className="card-title">System Status</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <StatusRow label="API Server" ok={health != null} value={health ? 'Online' : 'Offline'} />
            <StatusRow label="Database" ok={!error} value={!error ? 'Connected' : 'Error'} />
            <StatusRow label="CNN Model"    ok={modelInfo?.models?.[0]?.status === 'loaded'} value={modelInfo?.models?.[0]?.status === 'loaded' ? `v${modelInfo.models[0].version}` : 'Not trained'} />
            <StatusRow label="ResNet Model" ok={modelInfo?.models?.[1]?.status === 'loaded'} value={modelInfo?.models?.[1]?.status === 'loaded' ? `v${modelInfo.models[1].version}` : 'Not trained'} />
            <StatusRow label="RF Model"     ok={modelInfo?.models?.[2]?.status === 'loaded'} value={modelInfo?.models?.[2]?.status === 'loaded' ? `v${modelInfo.models[2].version}` : 'Not trained'} />
          </div>
          <div style={{ marginTop: 24 }}>
            <Link to="/analyze" className="btn btn-primary" style={{ width: '100%', justifyContent: 'center' }}>
              <Microscope size={18} />
              Analyze New Image
            </Link>
          </div>
        </div>
      </div>

      {/* Recent analyses */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
          <div className="card-title" style={{ margin: 0 }}>Recent Analyses</div>
          <Link to="/history" className="btn btn-ghost btn-sm">View All</Link>
        </div>
        {loading ? (
          <div>{[1,2,3].map(i => <div key={i} className="skeleton skeleton-card" style={{ marginBottom: 8 }} />)}</div>
        ) : history?.items?.length > 0 ? (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Analysis ID</th>
                  <th>Date</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                  <th>Model</th>
                </tr>
              </thead>
              <tbody>
                {history.items.map((item) => (
                  <tr key={item.analysis_id}>
                    <td style={{ fontFamily: 'monospace', fontSize: '0.8125rem', color: 'var(--accent-light)' }}>
                      {item.analysis_id.slice(0, 8)}...
                    </td>
                    <td style={{ color: 'var(--text-secondary)', fontSize: '0.8125rem' }}>
                      {new Date(item.created_at).toLocaleString()}
                    </td>
                    <td>
                      <span className={`badge badge-${item.prediction.toLowerCase()}`}>{item.prediction}</span>
                    </td>
                    <td style={{ fontSize: '0.875rem' }}>{(item.probability * 100).toFixed(1)}%</td>
                    <td style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>{item.model_name}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-state-icon">📊</div>
            <h3>No analyses yet</h3>
            <p>Upload an image to get started</p>
            <Link to="/analyze" className="btn btn-primary btn-sm" style={{ marginTop: 16 }}>Start Analysis</Link>
          </div>
        )}
      </div>

      {/* Disclaimer */}
      <div className="alert alert-warning">
        <AlertTriangle size={16} />
        <span>{DISCLAIMER}</span>
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, color }) {
  return (
    <div className="stat-card">
      <div className="stat-icon" style={{ background: `${color}15`, border: `1px solid ${color}30` }}>
        <span style={{ color }}>{icon}</span>
      </div>
      <div>
        <div className="stat-value" style={{ color }}>{value}</div>
        <div className="stat-label">{label}</div>
      </div>
    </div>
  );
}

function StatusRow({ label, ok, value }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>{label}</span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: '0.8125rem', color: ok ? 'var(--success)' : 'var(--danger)' }}>{value}</span>
        {ok ? <CheckCircle size={14} color="#10b981" /> : <XCircle size={14} color="#ef4444" />}
      </div>
    </div>
  );
}
