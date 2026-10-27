import React, { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard, ScanLine, History, BarChart2,
  Cpu, Info, ChevronLeft, ChevronRight, Heart,
  Activity, Wifi, WifiOff
} from 'lucide-react';
import apiService from '../services/api';
import './Sidebar.css';

const NAV_ITEMS = [
  { to: '/dashboard',   icon: LayoutDashboard, label: 'Dashboard'   },
  { to: '/analyze',     icon: ScanLine,        label: 'Analyze'     },
  { to: '/history',     icon: History,         label: 'History'     },
  { to: '/performance', icon: BarChart2,        label: 'Performance' },
  { to: '/models',      icon: Cpu,             label: 'Models'      },
  { to: '/about',       icon: Info,            label: 'About'       },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const [online, setOnline] = useState(null);
  const location = useLocation();

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        await apiService.getHealth();
        if (mounted) setOnline(true);
      } catch {
        if (mounted) setOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 15000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  return (
    <aside className={`sidebar${collapsed ? ' collapsed' : ''}`}>
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">
          <KidneyIcon size={collapsed ? 24 : 28} />
        </div>
        {!collapsed && (
          <div className="sidebar-logo-text">
            <span className="sidebar-brand">KidneyAI</span>
            <span className="sidebar-tagline">Explainable AI</span>
          </div>
        )}
        <button
          className="sidebar-toggle"
          onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}
            title={collapsed ? label : undefined}
          >
            <Icon size={20} className="sidebar-link-icon" />
            {!collapsed && <span className="sidebar-link-label">{label}</span>}
            {collapsed && (
              <span className="sidebar-tooltip">{label}</span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Bottom status */}
      <div className="sidebar-footer">
        {!collapsed ? (
          <div className="sidebar-status">
            <div className={`status-dot ${online === true ? 'online' : online === false ? 'offline' : 'checking'}`} />
            <span className="status-label">
              {online === null ? 'Checking...' : online ? 'API Online' : 'API Offline'}
            </span>
          </div>
        ) : (
          <div className="sidebar-status-icon" title={online === null ? 'Checking...' : online ? 'API Online' : 'API Offline'}>
            {online === true ? <Wifi size={16} color="#10b981" /> : <WifiOff size={16} color="#ef4444" />}
          </div>
        )}
      </div>
    </aside>
  );
}

function KidneyIcon({ size = 24 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M12 3C8.5 3 5 5.5 5 9.5C5 12.5 6.5 15 8 17C9.5 19 10 21 12 21C14 21 14.5 19 16 17C17.5 15 19 12.5 19 9.5C19 5.5 15.5 3 12 3Z"
        fill="url(#kidneyGrad)"
        opacity="0.9"
      />
      <path
        d="M12 3C9 3 7 5 7 8C7 10 8 12 8 12C9 10 10 9 12 9C14 9 15 10 16 12C16 12 17 10 17 8C17 5 15 3 12 3Z"
        fill="rgba(255,255,255,0.15)"
      />
      <defs>
        <linearGradient id="kidneyGrad" x1="5" y1="3" x2="19" y2="21" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#3b82f6" />
          <stop offset="100%" stopColor="#06b6d4" />
        </linearGradient>
      </defs>
    </svg>
  );
}
