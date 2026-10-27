import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { Upload, X, Image, AlertCircle, Loader, Info, FileImage, AlertTriangle } from 'lucide-react';
import apiService from '../services/api';
import { useToast } from '../components/ToastContext';

const MAX_SIZE_MB = 10;
const MAX_BYTES = MAX_SIZE_MB * 1024 * 1024;
const ALLOWED_TYPES = { 'image/jpeg': ['.jpg', '.jpeg'], 'image/png': ['.png'] };

export default function AnalyzePage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const toast = useToast();

  const onDrop = useCallback((accepted, rejected) => {
    setError(null);
    if (rejected.length > 0) {
      const msg = rejected[0].errors[0]?.message || 'Invalid file';
      setError(msg);
      return;
    }
    if (accepted.length > 0) {
      const f = accepted[0];
      if (f.size > MAX_BYTES) {
        setError(`File too large. Maximum size is ${MAX_SIZE_MB}MB.`);
        return;
      }
      setFile(f);
      const url = URL.createObjectURL(f);
      setPreview(url);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ALLOWED_TYPES,
    maxFiles: 1,
    maxSize: MAX_BYTES,
    multiple: false,
  });

  const clearFile = () => {
    setFile(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const result = await apiService.predictImage(file);
      toast.success(`Analysis complete: ${result.prediction}`);
      navigate(`/result/${result.analysis_id}`, { state: { result } });
    } catch (e) {
      setError(e.message);
      toast.error('Analysis failed: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container animate-fade-in">
      <div className="page-header">
        <h1 className="page-title">Image Analysis</h1>
        <p className="page-subtitle">Upload a kidney scan image for AI classification</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 24, alignItems: 'start' }}>
        {/* Upload area */}
        <div>
          {!preview ? (
            <div
              {...getRootProps()}
              className={`dropzone${isDragActive ? ' active' : ''}`}
            >
              <input {...getInputProps()} id="file-upload" aria-label="Upload kidney scan image" />
              <div className="dropzone-content">
                <div className="dropzone-icon">
                  <Upload size={48} />
                </div>
                <h3>{isDragActive ? 'Drop image here...' : 'Drag & Drop Image'}</h3>
                <p>or click to browse files</p>
                <div className="dropzone-formats">
                  <span className="badge badge-blue">JPG</span>
                  <span className="badge badge-blue">JPEG</span>
                  <span className="badge badge-blue">PNG</span>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>Max {MAX_SIZE_MB}MB</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="preview-container">
              <div className="preview-header">
                <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                  <FileImage size={16} style={{ display: 'inline', marginRight: 6 }} />
                  {file?.name}
                </span>
                <button className="btn btn-ghost btn-sm btn-icon" onClick={clearFile} aria-label="Remove image">
                  <X size={16} />
                </button>
              </div>
              <div className="preview-image-wrap">
                <img src={preview} alt="Uploaded kidney scan" className="preview-image" />
              </div>
              <div className="preview-meta">
                <span className="tag">Size: {(file?.size / 1024).toFixed(1)} KB</span>
                <span className="tag">Type: {file?.type}</span>
              </div>
            </div>
          )}

          {error && (
            <div className="alert alert-error" style={{ marginTop: 16 }}>
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          {preview && (
            <div style={{ display: 'flex', gap: 12, marginTop: 16 }}>
              <button
                className="btn btn-primary btn-lg"
                onClick={handleAnalyze}
                disabled={loading || !file}
                style={{ flex: 1, justifyContent: 'center' }}
                id="analyze-btn"
              >
                {loading ? (
                  <>
                    <div className="spinner" style={{ width: 20, height: 20 }} />
                    Analyzing...
                  </>
                ) : (
                  'Analyze Image'
                )}
              </button>
              <button className="btn btn-ghost btn-lg" onClick={clearFile} disabled={loading}>
                <X size={18} />
                Clear
              </button>
            </div>
          )}
        </div>

        {/* Info panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card">
            <div className="card-title">How Analysis Works</div>
            <ol style={{ paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 12, marginTop: 8 }}>
              {[
                'Image is validated (type, size, format)',
                'Preprocessed for each model pipeline',
                'Custom CNN classifies the image',
                'ResNet50 runs transfer learning inference',
                'Random Forest uses ResNet features',
                'Grad-CAM heatmap is generated',
                'Results displayed with probabilities',
              ].map((step, i) => (
                <li key={i} style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                  <span style={{ color: 'var(--accent-light)', fontWeight: 600, marginRight: 8 }}>{i + 1}.</span>
                  {step}
                </li>
              ))}
            </ol>
          </div>

          <div className="card">
            <div className="card-title">Output Classes</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 8 }}>
              {[
                { label: 'Cyst',   desc: 'Fluid-filled sac in kidney', color: 'badge-cyst' },
                { label: 'Normal', desc: 'No detected abnormality', color: 'badge-normal' },
                { label: 'Tumor',  desc: 'Abnormal tissue growth',     color: 'badge-tumor' },
              ].map((c) => (
                <div key={c.label} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span className={`badge ${c.color}`}>{c.label}</span>
                  <span style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>{c.desc}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="alert alert-warning">
            <AlertTriangle size={14} />
            <span style={{ fontSize: '0.8125rem' }}>
              For research and educational purposes only. Not a substitute for medical diagnosis.
            </span>
          </div>
        </div>
      </div>

      {/* Loading overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="spinner spinner-lg" />
          <h3 style={{ color: 'var(--text-primary)' }}>Analyzing image...</h3>
          <p style={{ color: 'var(--text-secondary)' }}>Running multi-model inference pipeline</p>
        </div>
      )}
    </div>
  );
}
