import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Brain, Layers, GitBranch, Eye, AlertTriangle, ChevronDown, Cpu, Zap, Shield } from 'lucide-react';
import './LandingPage.css';

export default function LandingPage() {
  const particlesRef = useRef(null);

  return (
    <div className="landing">
      {/* Hero */}
      <section className="hero">
        <div className="hero-bg" />
        <div className="particles" ref={particlesRef}>
          {Array.from({ length: 20 }).map((_, i) => (
            <div key={i} className="particle" style={{ '--delay': `${i * 0.3}s`, '--x': `${Math.random() * 100}%`, '--size': `${Math.random() * 4 + 2}px` }} />
          ))}
        </div>
        <div className="hero-content">
          <div className="hero-badge animate-fade-in">
            <span className="badge badge-blue">🔬 Research &amp; Education</span>
          </div>
          <h1 className="hero-title animate-slide-up">
            <span className="gradient-text">KidneyAI</span>
          </h1>
          <p className="hero-subtitle animate-slide-up">
            Explainable AI-Based Kidney Condition Analysis
          </p>
          <p className="hero-desc animate-slide-up">
            An educational research platform using deep learning and machine learning to analyze kidney medical images. Multi-model pipeline with Grad-CAM explainability.
          </p>
          <div className="hero-actions animate-slide-up">
            <Link to="/analyze" className="btn btn-primary btn-lg">
              <Zap size={20} />
              Start Analysis
              <ArrowRight size={18} />
            </Link>
            <Link to="/dashboard" className="btn btn-secondary btn-lg">
              View Dashboard
            </Link>
          </div>
          <div className="hero-scroll">
            <ChevronDown size={20} />
            <span>Scroll to explore</span>
          </div>
        </div>
        {/* Floating kidney visualization */}
        <div className="hero-visual animate-fade-in">
          <KidneyVisual />
        </div>
      </section>

      {/* How it works */}
      <section className="section">
        <div className="section-container">
          <div className="section-header">
            <span className="tag">Pipeline</span>
            <h2 className="section-title">How It Works</h2>
            <p className="section-desc">Three-step analysis pipeline powered by multi-model deep learning</p>
          </div>
          <div className="steps-grid">
            {[
              { step: '01', icon: '📤', title: 'Upload Image', desc: 'Drag and drop or select a kidney scan image (JPG or PNG). File validation ensures correct format.' },
              { step: '02', icon: '🧠', title: 'AI Analysis', desc: 'Three models run in parallel: Custom CNN, ResNet50 transfer learning, and Random Forest on deep features.' },
              { step: '03', icon: '📊', title: 'Explainable Results', desc: 'View prediction with confidence scores and Grad-CAM attention heatmap showing where the model focused.' },
            ].map((item) => (
              <div key={item.step} className="step-card glass">
                <div className="step-number">{item.step}</div>
                <div className="step-icon">{item.icon}</div>
                <h3 className="step-title">{item.title}</h3>
                <p className="step-desc">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology */}
      <section className="section section-alt">
        <div className="section-container">
          <div className="section-header">
            <span className="tag">Technology</span>
            <h2 className="section-title">Multi-Model Architecture</h2>
            <p className="section-desc">Three complementary approaches to kidney condition classification</p>
          </div>
          <div className="tech-grid">
            {[
              {
                icon: <Brain size={32} />,
                name: 'Custom CNN',
                color: '#3b82f6',
                items: ['4-layer Conv2D architecture', 'BatchNormalization', 'GlobalAveragePooling', '150×150 input', 'Data augmentation'],
              },
              {
                icon: <Layers size={32} />,
                name: 'ResNet50',
                color: '#06b6d4',
                items: ['ImageNet pretrained', 'Transfer learning', 'Frozen base + trainable head', '224×224 input', 'Fine-tuning support'],
              },
              {
                icon: <GitBranch size={32} />,
                name: 'Random Forest',
                color: '#10b981',
                items: ['ResNet50 feature extraction', '2048-dim feature vectors', 'scikit-learn ensemble', '200 estimators', 'Balanced class weights'],
              },
            ].map((tech) => (
              <div key={tech.name} className="tech-card glass">
                <div className="tech-icon" style={{ '--color': tech.color }}>
                  {tech.icon}
                </div>
                <h3 className="tech-name">{tech.name}</h3>
                <ul className="tech-list">
                  {tech.items.map((item) => (
                    <li key={item}><span className="tech-check">✓</span> {item}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Grad-CAM */}
      <section className="section">
        <div className="section-container">
          <div className="explainability-card glass">
            <div className="explainability-icon">
              <Eye size={40} />
            </div>
            <div className="explainability-content">
              <span className="tag">Explainable AI</span>
              <h2 className="explainability-title">Grad-CAM Visualization</h2>
              <p>
                Gradient-weighted Class Activation Mapping (Grad-CAM) generates heatmaps that highlight which regions of the image influenced the model's prediction. Available for CNN and ResNet50 models.
              </p>
              <div className="alert alert-warning" style={{ marginTop: '16px' }}>
                <AlertTriangle size={16} />
                <span>Model attention visualization — not a confirmed anatomical lesion location. Do not use for clinical decisions.</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Dataset */}
      <section className="section section-alt">
        <div className="section-container">
          <div className="section-header">
            <span className="tag">Dataset</span>
            <h2 className="section-title">Training Data</h2>
          </div>
          <div className="dataset-cards">
            {[
              { label: 'Cyst',   count: '3,709', color: '#f59e0b' },
              { label: 'Normal', count: '5,077', color: '#10b981' },
              { label: 'Tumor',  count: '2,283', color: '#ef4444' },
              { label: 'Total',  count: '11,069', color: '#3b82f6', highlight: true },
            ].map((d) => (
              <div key={d.label} className={`dataset-card glass ${d.highlight ? 'highlight' : ''}`} style={{ '--color': d.color }}>
                <div className="dataset-count" style={{ color: d.color }}>{d.count}</div>
                <div className="dataset-label">{d.label} Images</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="section">
        <div className="section-container">
          <div className="disclaimer-card">
            <div className="disclaimer-icon"><Shield size={32} /></div>
            <h3>Research Disclaimer</h3>
            <p>
              <strong>This application is intended for research and educational purposes only.</strong> It is not a clinically validated diagnostic tool and is NOT a substitute for professional medical diagnosis, clinical decision-making, or the advice of a qualified healthcare provider.
            </p>
            <p>
              All predictions are generated by experimental machine learning models trained on a publicly available dataset. Model probabilities are uncalibrated and should not be interpreted as clinical confidence levels.
            </p>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section cta-section">
        <div className="section-container">
          <div className="cta-content glass">
            <h2 className="gradient-text">Ready to Explore?</h2>
            <p>Upload a kidney scan image to run the AI analysis pipeline.</p>
            <Link to="/analyze" className="btn btn-primary btn-lg">
              <Zap size={20} />
              Analyze an Image
              <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <strong className="gradient-text">KidneyAI</strong>
            <span> — Research &amp; Educational Platform</span>
          </div>
          <p className="footer-disclaimer">
            For research and educational purposes only. Not a medical device or clinical diagnostic tool.
          </p>
        </div>
      </footer>
    </div>
  );
}

function KidneyVisual() {
  return (
    <div className="kidney-visual">
      <svg viewBox="0 0 200 240" fill="none" xmlns="http://www.w3.org/2000/svg" className="kidney-svg">
        <defs>
          <linearGradient id="kg1" x1="40" y1="20" x2="160" y2="220" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.8"/>
            <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.6"/>
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        {/* Outer kidney shape */}
        <ellipse cx="100" cy="120" rx="65" ry="90" fill="url(#kg1)" filter="url(#glow)" opacity="0.3"/>
        <path d="M100 35 C65 35, 40 65, 40 100 C40 135, 60 165, 80 185 C90 195, 100 200, 100 200 C100 200, 110 195, 120 185 C140 165, 160 135, 160 100 C160 65, 135 35, 100 35 Z"
          fill="url(#kg1)" opacity="0.6"/>
        {/* Inner detail */}
        <path d="M100 55 C80 55, 65 75, 65 100 C65 115, 70 130, 80 145" stroke="rgba(255,255,255,0.3)" strokeWidth="2" fill="none"/>
        {/* Scanning line */}
        <line x1="40" y1="120" x2="160" y2="120" stroke="rgba(59,130,246,0.8)" strokeWidth="1.5" strokeDasharray="4,4" className="scan-line"/>
        {/* Grid overlay */}
        {[60,80,100,120,140].map(y => (
          <line key={y} x1="40" y1={y} x2="160" y2={y} stroke="rgba(59,130,246,0.1)" strokeWidth="0.5"/>
        ))}
        {[60,80,100,120,140].map(x => (
          <line key={x} x1={x} y1="35" x2={x} y2="200" stroke="rgba(59,130,246,0.1)" strokeWidth="0.5"/>
        ))}
      </svg>
      <div className="kidney-glow" />
    </div>
  );
}
