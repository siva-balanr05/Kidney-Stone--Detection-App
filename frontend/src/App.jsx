import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import { ToastProvider } from './components/ToastContext';
import './App.css';

const LandingPage      = lazy(() => import('./pages/LandingPage'));
const Dashboard        = lazy(() => import('./pages/Dashboard'));
const AnalyzePage      = lazy(() => import('./pages/AnalyzePage'));
const ResultPage       = lazy(() => import('./pages/ResultPage'));
const HistoryPage      = lazy(() => import('./pages/HistoryPage'));
const PerformancePage  = lazy(() => import('./pages/PerformancePage'));
const ModelsPage       = lazy(() => import('./pages/ModelsPage'));
const AboutPage        = lazy(() => import('./pages/AboutPage'));

function Fallback() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: 'var(--bg)' }}>
      <div className="spinner spinner-lg" />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <Suspense fallback={<Fallback />}>
          <Routes>
            {/* Landing — no sidebar */}
            <Route path="/" element={<LandingPage />} />

            {/* App pages — with sidebar layout */}
            <Route path="/dashboard"   element={<Layout><Dashboard /></Layout>} />
            <Route path="/analyze"     element={<Layout><AnalyzePage /></Layout>} />
            <Route path="/result/:id"  element={<Layout><ResultPage /></Layout>} />
            <Route path="/history"     element={<Layout><HistoryPage /></Layout>} />
            <Route path="/performance" element={<Layout><PerformancePage /></Layout>} />
            <Route path="/models"      element={<Layout><ModelsPage /></Layout>} />
            <Route path="/about"       element={<Layout><AboutPage /></Layout>} />

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </ToastProvider>
    </BrowserRouter>
  );
}
