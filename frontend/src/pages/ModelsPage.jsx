import React, { useEffect, useState } from 'react';
import { AlertTriangle, Cpu, Brain, GitBranch, Layers } from 'lucide-react';
import apiService from '../services/api';

const ARCHITECTURES = [
  {
    key: 'cnn',
    icon: <Brain size={24} />,
    color: '#3b82f6',
    title: 'Custom CNN',
    pipeline: [
      'Input: 150 × 150 × 3',
      'Conv2D(32) + BatchNorm + MaxPool',
      'Conv2D(64) + BatchNorm + MaxPool',
      'Conv2D(128) + BatchNorm + MaxPool',
      'Conv2D(256) + BatchNorm + GlobalAveragePooling',
      'Dense(256, ReLU) + Dropout(0.5)',
      'Output: Softmax(3 classes)',
    ],
    notes: 'Trained from scratch. Data augmentation (flip, brightness, contrast). Early stopping + ModelCheckpoint.',
  },
  {
    key: 'resnet',
    icon: <Layers size={24} />,
    color: '#06b6d4',
    title: 'ResNet50 Transfer Learning',
    pipeline: [
      'Input: 224 × 224 × 3',
      'ResNet50 base (ImageNet pretrained, frozen)',
      'GlobalAveragePooling2D',
      'Dense(256, ReLU) + Dropout(0.4)',
      'Output: Softmax(3 classes)',
      'Phase 2: Fine-tune last 30 base layers (lr=1e-5)',
    ],
    notes: 'Two-phase training: frozen head first, then fine-tuning. ImageNet preprocessing applied.',
  },
  {
    key: 'rf',
    icon: <GitBranch size={24} />,
    color: '#10b981',
    title: 'ResNet50 + Random Forest',
    pipeline: [
      'Input: 224 × 224 × 3',
      'ResNet50 feature extractor (no top)',
      'GlobalAveragePooling2D → 2048-dim vector',
      'StandardScaler (fit on train only)',
      'RandomForestClassifier(n_estimators=200)',
      'Output: class label',
    ],
    notes: 'Hybrid model. No Grad-CAM for RF itself; use CNN/ResNet for explainability.',
  },
];

export default function ModelsPage() {
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

  const getStatus = (key) => {
    if (!modelInfo?.models) return null;
    const idx = { cnn: 0, resnet: 1, rf: 2 }[key];
    return modelInfo.models[idx];
  };

  return (
    <div className="page-container animate-fade-in">
      <div className="page-header">
        <h1 className="page-title">Model Information</h1>
        <p className="page-subtitle">Architecture details and availability status of all three models</p>
      </div>

      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {[1,2,3].map(i => <div key={i} className="skeleton skeleton-card" style={{ height: 220 }} />)}
        </div>
      )}

      {error && (
        <div className="alert alert-error" style={{ marginBottom: 16 }}>
          <AlertTriangle size={16} />
          <span>Could not load model info: {error}</span>
        </div>
      )}

      {ARCHITECTURES.map((arch) => {
        const status = getStatus(arch.key);
        return (
          <div key={arch.key} className="card" style={{ marginBottom: 20, borderTop: `3px solid ${arch.color}` }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ color: arch.color, background: `${arch.color}15`, padding: 10, borderRadius: 10 }}>
                  {arch.icon}
                </div>
                <div>
                  <div className="card-title" style={{ margin: 0 }}>{arch.title}</div>
                  {status && (
                    <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: 2 }}>
                      {status.framework} · Input: {status.input_size} · {status.num_classes} classes
                    </div>
                  )}
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
                {status ? (
                  <>
                    <span className={`badge ${status.status === 'loaded' ? 'badge-normal' : 'badge-red'}`}>
                      {status.status === 'loaded' ? '✓ Loaded' : '✗ Not Trained'}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      v{status.version}
                    </span>
                  </>
                ) : (
                  <span className="badge badge-red">—</span>
                )}
              </div>
            </div>

            {/* Architecture pipeline */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Architecture Pipeline
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
                {arch.pipeline.map((step, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: arch.color, flexShrink: 0 }} />
                      {i < arch.pipeline.length - 1 && (
                        <div style={{ width: 2, height: 18, background: `${arch.color}40` }} />
                      )}
                    </div>
                    <div style={{
                      fontSize: '0.8125rem',
                      color: 'var(--text-primary)',
                      fontFamily: 'monospace',
                      background: 'var(--bg)',
                      border: '1px solid var(--border)',
                      borderRadius: 6,
                      padding: '4px 10px',
                      marginBottom: i < arch.pipeline.length - 1 ? 0 : 0,
                    }}>
                      {step}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Notes */}
            <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', background: 'var(--bg)', borderRadius: 8, padding: '10px 14px', borderLeft: `3px solid ${arch.color}` }}>
              {arch.notes}
            </div>

            {/* Accuracy badge if available */}
            {status?.metrics_available && status?.accuracy != null && (
              <div style={{ marginTop: 12 }}>
                <span className="badge badge-blue">
                  Test Accuracy: {(status.accuracy * 100).toFixed(1)}%
                </span>
              </div>
            )}

            {/* Not trained notice */}
            {status && status.status !== 'loaded' && (
              <div className="alert alert-warning" style={{ marginTop: 12 }}>
                <AlertTriangle size={14} />
                <span style={{ fontSize: '0.8125rem' }}>
                  Model not trained. Run the training script: <code>python -m ml.train_{arch.key === 'rf' ? 'random_forest' : arch.key}</code>
                </span>
              </div>
            )}
          </div>
        );
      })}

      <div className="alert alert-warning">
        <AlertTriangle size={14} />
        <span style={{ fontSize: '0.8125rem' }}>
          For research and educational purposes only. Do not use model outputs for clinical diagnosis.
        </span>
      </div>
    </div>
  );
}
