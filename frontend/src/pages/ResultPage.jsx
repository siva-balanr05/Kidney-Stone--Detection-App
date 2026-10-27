import React, { useEffect, useState } from 'react';
import { useParams, useLocation, Link, useNavigate } from 'react-router-dom';
import { AlertTriangle, ArrowLeft, CheckCircle, Eye, Cpu, BarChart2 } from 'lucide-react';
import apiService from '../services/api';

const CLASS_COLORS = { Cyst: '#f59e0b', Normal: '#10b981', Tumor: '#ef4444' };
const CLASS_BADGE  = { Cyst: 'badge-cyst', Normal: 'badge-normal', Tumor: 'badge-tumor' };

export default function ResultPage() {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const [result, setResult] = useState(location.state?.result || null);
  const [loading, setLoading] = useState(!result);
  const [error, setError]   = useState(null);

  useEffect(() => {
    if (result) return;
    (async () => {
      try {
        const data = await apiService.getAnalysis(id);
        setResult(data);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    })();
  }, [id]);

  if (loading) return (
    <div className="page-container animate-fade-in" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 400 }}>
      <div className="spinner spinner-lg" />
    </div>
  );

  if (error) return (
    <div className="page-container animate-fade-in">
      <div className="alert alert-error">
        <AlertTriangle size={16} />
        <span>{error}</span>
      </div>
      <button className="btn btn-ghost" style={{ marginTop: 16 }} onClick={() => navigate('/analyze')}>
        <ArrowLeft size={16} /> Back to Analyze
      </button>
    </div>
  );

  if (!result) return null;

  const probs = result.all_probabilities || {};
  const classes = ['Cyst', 'Normal', 'Tumor'];

  return (
    <div className="page-container animate-fade-in">
      <div className="page-header" style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <Link to="/analyze" className="btn btn-ghost btn-sm btn-icon" aria-label="Back">
          <ArrowLeft size={18} />
        </Link>
        <div>
          <h1 className="page-title">Analysis Result</h1>
          <p className="page-subtitle" style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
            ID: {result.analysis_id}
          </p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>

        {/* LEFT COLUMN */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

          {/* Prediction */}
          <div className="card">
            <div className="card-title">Prediction</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 12 }}>
              <CheckCircle size={32} color={CLASS_COLORS[result.prediction] || '#3b82f6'} />
              <div>
                <span className={`badge ${CLASS_BADGE[result.prediction] || 'badge-blue'}`}
                      style={{ fontSize: '1.1rem', padding: '6px 16px' }}>
                  {result.prediction}
                </span>
                <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: 6 }}>
                  Primary classification result
                </div>
              </div>
            </div>
          </div>

          {/* Probability bars */}
          <div className="card">
            <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <BarChart2 size={16} /> Class Probabilities
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginTop: 12 }}>
              {classes.map(cls => {
                const p = probs[cls] ?? 0;
                return (
                  <div key={cls}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span className={`badge ${CLASS_BADGE[cls]}`}>{cls}</span>
                      <span style={{ fontSize: '0.875rem', fontWeight: 600, color: CLASS_COLORS[cls] }}>
                        {(p * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ height: 8, background: 'var(--border)', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{
                        height: '100%',
                        width: `${(p * 100).toFixed(1)}%`,
                        background: CLASS_COLORS[cls],
                        borderRadius: 4,
                        transition: 'width 0.6s ease'
                      }} />
                    </div>
                  </div>
                );
              })}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 12 }}>
              ⚠ Probabilities are uncalibrated. Do not interpret as clinical confidence.
            </div>
          </div>

          {/* Model info */}
          <div className="card">
            <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Cpu size={16} /> Model Information
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 12 }}>
              {[
                ['Model', result.model_name],
                ['Version', result.model_version],
                ['Analysis ID', result.analysis_id?.slice(0, 16) + '…'],
                ['Grad-CAM', result.gradcam_available ? 'Available' : 'Not available'],
              ].map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>{k}</span>
                  <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{v}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Warnings */}
          {result.warnings?.length > 0 && (
            <div className="alert alert-warning">
              <AlertTriangle size={14} />
              <div>
                {result.warnings.map((w, i) => <div key={i}>{w}</div>)}
              </div>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

          {/* Grad-CAM */}
          <div className="card">
            <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Eye size={16} /> Grad-CAM Explainability
            </div>
            {result.gradcam_available && result.gradcam_image_b64 ? (
              <div style={{ marginTop: 12 }}>
                <img
                  src={`data:image/jpeg;base64,${result.gradcam_image_b64}`}
                  alt="Grad-CAM heatmap"
                  style={{ width: '100%', borderRadius: 8, border: '1px solid var(--border)' }}
                />
                <div className="alert alert-warning" style={{ marginTop: 12 }}>
                  <AlertTriangle size={14} />
                  <span style={{ fontSize: '0.8125rem' }}>
                    Model attention visualization — not a confirmed anatomical lesion location. Do not use for clinical decisions.
                  </span>
                </div>
              </div>
            ) : (
              <div className="empty-state" style={{ padding: '32px 0' }}>
                <div className="empty-state-icon">🔍</div>
                <p>Grad-CAM not available</p>
                <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                  {result.model_name?.includes('Random Forest')
                    ? 'Grad-CAM is not applicable to Random Forest models.'
                    : 'Train a CNN or ResNet model to enable Grad-CAM visualization.'}
                </p>
              </div>
            )}
          </div>

          {/* Disclaimer */}
          <div className="disclaimer-card" style={{ background: 'rgba(239,68,68,0.05)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 12, padding: 20 }}>
            <div style={{ display: 'flex', gap: 12 }}>
              <AlertTriangle size={20} color="#ef4444" style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 8 }}>
                  Research Disclaimer
                </div>
                <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
                  {result.disclaimer || 'This result is for research and educational purposes only. It is not a substitute for professional medical diagnosis or clinical decision-making.'}
                </p>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div style={{ display: 'flex', gap: 12 }}>
            <Link to="/analyze" className="btn btn-primary" style={{ flex: 1, justifyContent: 'center' }}>
              Analyze Another Image
            </Link>
            <Link to="/history" className="btn btn-ghost" style={{ flex: 1, justifyContent: 'center' }}>
              View History
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
