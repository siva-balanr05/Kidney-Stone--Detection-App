import React, { useEffect, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Search, Trash2, AlertTriangle, RefreshCw, ChevronLeft, ChevronRight, Filter } from 'lucide-react';
import apiService from '../services/api';
import { useToast } from '../components/ToastContext';

const CLASS_BADGE = { Cyst: 'badge-cyst', Normal: 'badge-normal', Tumor: 'badge-tumor' };
const CLASSES = ['All', 'Cyst', 'Normal', 'Tumor'];
const LIMIT = 20;

export default function HistoryPage() {
  const toast = useToast();
  const [items, setItems]       = useState([]);
  const [total, setTotal]       = useState(0);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);
  const [skip, setSkip]         = useState(0);
  const [filter, setFilter]     = useState('All');
  const [search, setSearch]     = useState('');
  const [deleting, setDeleting] = useState(null);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = { skip, limit: LIMIT };
      if (filter !== 'All') params.prediction = filter;
      const data = await apiService.getHistory(params);
      setItems(data.items || []);
      setTotal(data.total || 0);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [skip, filter]);

  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this analysis record?')) return;
    setDeleting(id);
    try {
      await apiService.deleteAnalysis(id);
      toast.success('Analysis deleted.');
      fetchHistory();
    } catch (e) {
      toast.error('Delete failed: ' + e.message);
    } finally {
      setDeleting(null);
    }
  };

  const displayed = search
    ? items.filter(i => i.analysis_id.toLowerCase().includes(search.toLowerCase()))
    : items;

  const totalPages = Math.ceil(total / LIMIT);
  const currentPage = Math.floor(skip / LIMIT) + 1;

  return (
    <div className="page-container animate-fade-in">
      <div className="page-header">
        <h1 className="page-title">Analysis History</h1>
        <p className="page-subtitle">Browse and manage past kidney scan analyses</p>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap', alignItems: 'center' }}>
        {/* Search */}
        <div style={{ position: 'relative', flex: 1, minWidth: 200 }}>
          <Search size={16} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
          <input
            type="text"
            placeholder="Search by analysis ID..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{
              width: '100%', paddingLeft: 34, paddingRight: 12, height: 38,
              background: 'var(--surface)', border: '1px solid var(--border)',
              borderRadius: 8, color: 'var(--text-primary)', fontSize: '0.875rem',
              outline: 'none', boxSizing: 'border-box'
            }}
          />
        </div>

        {/* Filter */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Filter size={16} color="var(--text-muted)" />
          {CLASSES.map(c => (
            <button
              key={c}
              className={`btn btn-sm ${filter === c ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => { setFilter(c); setSkip(0); }}
            >
              {c}
            </button>
          ))}
        </div>

        {/* Refresh */}
        <button className="btn btn-ghost btn-sm btn-icon" onClick={fetchHistory} title="Refresh">
          <RefreshCw size={16} />
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="alert alert-error" style={{ marginBottom: 16 }}>
          <AlertTriangle size={16} />
          <span>{error}</span>
          <button className="btn btn-ghost btn-sm" onClick={fetchHistory} style={{ marginLeft: 'auto' }}>Retry</button>
        </div>
      )}

      {/* Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        {loading ? (
          <div style={{ padding: 24 }}>
            {[1,2,3,4,5].map(i => (
              <div key={i} className="skeleton skeleton-card" style={{ marginBottom: 12, height: 48 }} />
            ))}
          </div>
        ) : displayed.length === 0 ? (
          <div className="empty-state" style={{ padding: '60px 0' }}>
            <div className="empty-state-icon">📋</div>
            <h3>No analyses found</h3>
            <p>{search ? 'No records match your search.' : 'Upload an image to create your first analysis.'}</p>
            <Link to="/analyze" className="btn btn-primary btn-sm" style={{ marginTop: 16 }}>Start Analysis</Link>
          </div>
        ) : (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Analysis ID</th>
                  <th>Date &amp; Time</th>
                  <th>Prediction</th>
                  <th>Probability</th>
                  <th>Model</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {displayed.map(item => (
                  <tr key={item.analysis_id}>
                    <td style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'var(--accent-light)' }}>
                      {item.analysis_id.slice(0, 12)}…
                    </td>
                    <td style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                      {new Date(item.created_at).toLocaleString()}
                    </td>
                    <td>
                      <span className={`badge ${CLASS_BADGE[item.prediction] || 'badge-blue'}`}>
                        {item.prediction}
                      </span>
                    </td>
                    <td style={{ fontSize: '0.875rem', fontWeight: 600 }}>
                      {(item.probability * 100).toFixed(1)}%
                    </td>
                    <td style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                      {item.model_name}
                    </td>
                    <td>
                      <button
                        className="btn btn-ghost btn-sm btn-icon"
                        onClick={() => handleDelete(item.analysis_id)}
                        disabled={deleting === item.analysis_id}
                        title="Delete"
                        style={{ color: 'var(--danger)' }}
                      >
                        {deleting === item.analysis_id
                          ? <div className="spinner" style={{ width: 14, height: 14 }} />
                          : <Trash2 size={14} />}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Pagination */}
      {total > LIMIT && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 16 }}>
          <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
            Page {currentPage} of {totalPages} ({total} total)
          </span>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              className="btn btn-ghost btn-sm"
              disabled={skip === 0}
              onClick={() => setSkip(s => Math.max(0, s - LIMIT))}
            >
              <ChevronLeft size={16} /> Prev
            </button>
            <button
              className="btn btn-ghost btn-sm"
              disabled={skip + LIMIT >= total}
              onClick={() => setSkip(s => s + LIMIT)}
            >
              Next <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="alert alert-warning" style={{ marginTop: 24 }}>
        <AlertTriangle size={14} />
        <span style={{ fontSize: '0.8125rem' }}>
          For research and educational purposes only. No sensitive patient data is stored.
        </span>
      </div>
    </div>
  );
}
