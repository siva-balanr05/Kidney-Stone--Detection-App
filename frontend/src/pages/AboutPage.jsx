import React, { useState } from 'react';
import { Brain, Layers, GitBranch, Eye, AlertTriangle, Database, Shield, BookOpen, ChevronDown, ChevronUp } from 'lucide-react';

function Section({ title, icon, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
      >
        <div className="card-title" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ color: 'var(--accent-light)' }}>{icon}</span>
          {title}
        </div>
        {open ? <ChevronUp size={18} color="var(--text-muted)" /> : <ChevronDown size={18} color="var(--text-muted)" />}
      </button>
      {open && <div style={{ marginTop: 16, color: 'var(--text-secondary)', lineHeight: 1.75, fontSize: '0.9rem' }}>{children}</div>}
    </div>
  );
}

export default function AboutPage() {
  return (
    <div className="page-container animate-fade-in">
      <div className="page-header">
        <h1 className="page-title">About KidneyAI</h1>
        <p className="page-subtitle">Research platform — explainable AI for kidney condition image analysis</p>
      </div>

      <Section title="Project Objective" icon={<BookOpen size={18} />} defaultOpen>
        <p>
          KidneyAI is a research and educational platform that applies deep learning and
          machine learning techniques to classify kidney CT scan images into three categories:
          <strong> Cyst</strong>, <strong>Normal</strong>, and <strong>Tumor</strong>.
        </p>
        <p style={{ marginTop: 10 }}>
          The goal is to explore how multiple model architectures — a custom CNN, ResNet50
          transfer learning, and a ResNet50 + Random Forest hybrid — compare on the same
          classification task, while providing Grad-CAM explainability to understand model attention.
        </p>
      </Section>

      <Section title="Custom CNN Architecture" icon={<Brain size={18} />}>
        <p>
          A <strong>Convolutional Neural Network (CNN)</strong> learns spatial features
          directly from raw pixel values.
        </p>
        <ul style={{ paddingLeft: 20, marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <li><strong>Convolution:</strong> Small filters (kernels) slide across the image to detect local patterns — edges, textures, shapes.</li>
          <li><strong>ReLU:</strong> Activation function that keeps positive values and zeros negatives — adds non-linearity so the model learns complex patterns.</li>
          <li><strong>Batch Normalization:</strong> Normalizes layer outputs to stabilize and speed up training.</li>
          <li><strong>MaxPooling:</strong> Reduces spatial dimensions (downsampling) by keeping the strongest activation in each patch.</li>
          <li><strong>GlobalAveragePooling:</strong> Averages each feature map into a single value — compact representation, reduces overfitting vs. Flatten.</li>
          <li><strong>Dense + Dropout:</strong> Fully connected layer learns class-specific combinations. Dropout randomly disables neurons during training to prevent overfitting.</li>
          <li><strong>Softmax output:</strong> Converts raw scores to probabilities that sum to 1.0 across 3 classes.</li>
        </ul>
      </Section>

      <Section title="ResNet50 Transfer Learning" icon={<Layers size={18} />}>
        <p>
          <strong>ResNet50</strong> is a 50-layer deep neural network pre-trained on ImageNet
          (1.2M images, 1000 classes). Transfer learning reuses this learned knowledge.
        </p>
        <ul style={{ paddingLeft: 20, marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <li><strong>Residual connections:</strong> Skip connections allow gradients to flow directly through the network, enabling training of very deep networks without vanishing gradients.</li>
          <li><strong>Frozen base:</strong> In Phase 1, all ResNet50 weights are frozen. Only the new classification head (Dense + Dropout + Softmax) is trained.</li>
          <li><strong>Fine-tuning:</strong> In Phase 2, the last 30 layers of ResNet50 are unfrozen and trained with a very low learning rate (1e-5) to adapt ImageNet features to kidney images.</li>
        </ul>
        <div className="alert alert-warning" style={{ marginTop: 10 }}>
          <AlertTriangle size={13} />
          <span style={{ fontSize: '0.8125rem' }}>ImageNet pre-training does not guarantee good medical image performance. Always evaluate on your actual dataset.</span>
        </div>
      </Section>

      <Section title="Random Forest Hybrid" icon={<GitBranch size={18} />}>
        <p>
          The <strong>Random Forest</strong> model uses ResNet50 as a feature extractor rather
          than a classifier. The 2048-dimensional feature vector from ResNet50's global average
          pooling layer is fed into an ensemble of decision trees.
        </p>
        <ul style={{ paddingLeft: 20, marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <li><strong>Feature extraction:</strong> ResNet50 (without its classification head) converts each image into a 2048-dim vector capturing semantic features.</li>
          <li><strong>StandardScaler:</strong> Normalizes feature vectors (zero mean, unit variance) — fit only on training data to prevent data leakage.</li>
          <li><strong>Random Forest:</strong> 200 decision trees, each trained on a bootstrap sample with random feature subsets. Final prediction = majority vote.</li>
          <li><strong>Grad-CAM limitation:</strong> Grad-CAM is gradient-based and requires a differentiable neural network. It cannot be applied directly to Random Forest.</li>
        </ul>
      </Section>

      <Section title="Grad-CAM Explainability" icon={<Eye size={18} />}>
        <p>
          <strong>Gradient-weighted Class Activation Mapping (Grad-CAM)</strong> produces
          heatmaps showing which image regions most influenced the model's prediction.
        </p>
        <ul style={{ paddingLeft: 20, marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <li>Gradients of the predicted class score are computed with respect to the last convolutional layer's output.</li>
          <li>These gradients are global-average-pooled to get neuron importance weights.</li>
          <li>A weighted combination of feature maps is computed, then ReLU-activated and resized to input dimensions.</li>
          <li>The heatmap is superimposed on the original image.</li>
        </ul>
        <div className="alert alert-warning" style={{ marginTop: 10 }}>
          <AlertTriangle size={13} />
          <span style={{ fontSize: '0.8125rem' }}>Grad-CAM shows model attention, NOT confirmed anatomical lesion locations. It is an explanation of the model, not a clinical localization tool.</span>
        </div>
      </Section>

      <Section title="Dataset" icon={<Database size={18} />}>
        <p>
          Training data is sourced from the publicly available <strong>CT Kidney Dataset</strong>
          on Kaggle. The dataset contains labelled kidney CT scan images across three categories:
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginTop: 12 }}>
          {[
            { label: 'Cyst',   count: '3,709', color: '#f59e0b' },
            { label: 'Normal', count: '5,077', color: '#10b981' },
            { label: 'Tumor',  count: '2,283', color: '#ef4444' },
            { label: 'Total',  count: '11,069', color: '#3b82f6' },
          ].map(d => (
            <div key={d.label} style={{ textAlign: 'center', background: 'var(--bg)', borderRadius: 8, padding: '12px 8px', border: `1px solid ${d.color}30` }}>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, color: d.color }}>{d.count}</div>
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>{d.label}</div>
            </div>
          ))}
        </div>
        <p style={{ marginTop: 12 }}>
          Patient-level split information is unavailable in this dataset. Images are split randomly (70% train / 15% val / 15% test). This is a limitation — see below.
        </p>
      </Section>

      <Section title="Limitations" icon={<AlertTriangle size={18} />}>
        <ul style={{ paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <li><strong>Dataset:</strong> No patient-level split is possible — images from the same patient may appear in both training and test sets, inflating reported performance.</li>
          <li><strong>Generalization:</strong> Models trained on this dataset may not generalize to images from different scanners, imaging protocols, or populations.</li>
          <li><strong>Probability calibration:</strong> Model output probabilities are uncalibrated and should not be interpreted as clinical confidence levels.</li>
          <li><strong>Grad-CAM:</strong> Attention heatmaps explain the model's behaviour, not true lesion anatomy. They are not validated for clinical use.</li>
          <li><strong>Clinical scope:</strong> The model is not trained on or validated for stone-specific detection. The classes are Cyst, Normal, and Tumor.</li>
          <li><strong>Missing stone class:</strong> Note: this version does not include a "Stone" class — the dataset used has Cyst, Normal, and Tumor.</li>
        </ul>
      </Section>

      <Section title="Ethical Considerations" icon={<Shield size={18} />}>
        <ul style={{ paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <li>No personally identifiable patient information is collected, stored, or displayed.</li>
          <li>All predictions are from experimental models and are explicitly marked as non-clinical.</li>
          <li>The system must not be used for actual medical screening or diagnosis.</li>
          <li>Model limitations and uncertainty are communicated at every prediction step.</li>
          <li>This application complies with research and educational use only. Any clinical deployment requires regulatory approval, clinical validation, and appropriate oversight.</li>
        </ul>
      </Section>

      {/* Disclaimer */}
      <div style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.25)', borderRadius: 12, padding: 24, marginTop: 8 }}>
        <div style={{ display: 'flex', gap: 12 }}>
          <Shield size={24} color="#ef4444" style={{ flexShrink: 0 }} />
          <div>
            <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 8, fontSize: '1rem' }}>
              Research &amp; Educational Disclaimer
            </div>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.7, margin: 0 }}>
              This application is intended for research and educational purposes only. It is not a clinically validated
              diagnostic tool and is <strong>NOT</strong> a substitute for professional medical diagnosis, clinical
              decision-making, or the advice of a qualified healthcare provider. All predictions are generated by
              experimental machine learning models trained on a publicly available dataset.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
