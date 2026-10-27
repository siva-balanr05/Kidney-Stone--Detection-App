import React, { useEffect, useState } from 'react';
import { AlertTriangle, BarChart2, Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import apiService from '../services/api';

const METRIC_LABELS = {
  accuracy:  'Accuracy',
  precision: 'Precision',
  recall:    'Recall',
  f1:        'F1-Score',
};

const MODEL_COLORS = ['#3b82f6', '#06b6d4', '#10b981'];

function MetricRow({ label, value }) {
  const pct = value != null ? (value * 100).toFixed(1) : null;
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
      <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>{label}</span>
      {pct != null ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 100, height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${pct}%`, background: 'var(--accent)', borderRadius: 3 }} />
          </div>
          <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', minWidth: 44 }}>{pct}%</span>
        </div>
      ) : (
        <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>—</span>
      )}
    </div>
  );
}

export default function PerformancePage() {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const data = await apiService.getModelInfo();
        setModelInfo(data);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // Build chart data from available model metrics
  const chartData = React.useMemo(() => {
    if (!modelInfo?.models) return [];
    const metrics = ['accuracy', 'precision', 'recall', 'f1'];
    return metrics.map(m => {
      const row = { metric: METRIC_LABELS[m] };
      modelInfo.models.forEach(model => {
        if (model.metrics_available && model[m] != null) {
          row[model.name] = parseFloat((model[m] * 100).toFixed(1));
        }
      });
      return row;
    });
  }, [modelInfo]);

  const hasAnyMetrics = modelInfo?.models?.some(m => m.metrics_available);

  return (
    <div className="page-container animate-fade-in">
      <div className="page-header">
        <h1 className="page-title">Model Performance</h1>
        <p className="page-subtitle">Evaluation metrics from the test dataset</p>
      </div>

      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {[1,2,3].map(i => <div key={i} className="skeleton skeleton-card" style={{ height: 200 }} />)}
        </div>
      )}

      {error && (
        <div className="alert alert-error">
          <AlertTriangle size={16} />
          <span>Cannot load model info: {error} — make sure the backend is running.</span>
        </div>
      )}

      {/* Per-model metric cards */}
      {!loading && modelInfo?.models?.map((model, idx) => (
        <div key={model.name} className="card" style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <div>
              <div className="card-title" style={{ margin: 0, color: MODEL_COLORS[idx] }}>{model.name}</div>
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: 4 }}>{model.model_type} · {model.framework}</div>
            </div>
            <span className={`badge ${model.status === 'loaded' ? 'badge-normal' : 'badge-red'}`}>
              {model.status === 'loaded' ? '✓ Loaded' : '✗ Not Trained'}
            </span>
          </div>

          {model.metrics_available ? (
            <div>
              <MetricRow label="Accuracy"  value={model.accuracy} />
              <MetricRow label="Precision" value={model.precision} />
              <MetricRow label="Recall"    value={model.recall} />
              <MetricRow label="F1-Score"  value={model.f1} />
            </div>
          ) : (
            <div className="alert alert-warning" style={{ marginTop: 8 }}>
              <Info size={14} />
              <span style={{ fontSize: '0.8125rem' }}>
                Evaluation not available — train and evaluate the model first by running <code>python -m ml.train_cnn</code> (or resnet / random_forest).
              </span>
            </div>
          )}
        </div>
      ))}

      {/* Comparison chart */}
      {!loading && hasAnyMetrics && chartData.length > 0 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <BarChart2 size={16} /> Model Comparison Chart
          </div>
          <div style={{ marginTop: 8, fontSize: '0.8125rem', color: 'var(--text-muted)', marginBottom: 16 }}>
            Metrics evaluated on the held-out test set (15% of dataset, same split across all models).
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="metric" tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
              <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
              <Tooltip formatter={(v) => `${v}%`} contentStyle={{ background: 'var(--surface-elevated)', border: '1px solid var(--border)', borderRadius: 8 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {modelInfo?.models?.map((m, i) => m.metrics_available && (
                <Bar key={m.name} dataKey={m.name} fill={MODEL_COLORS[i]} radius={[4,4,0,0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Disclaimer */}
      <div className="alert alert-warning">
        <AlertTriangle size={14} />
        <span style={{ fontSize: '0.8125rem' }}>
          Metrics are measured on the training dataset split. Performance on external datasets may differ. Not validated for clinical use.
        </span>
      </div>
    </div>
  );
}
