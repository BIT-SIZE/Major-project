import React, { useState, useEffect } from 'react'
import { NavLink } from 'react-router-dom'
import { 
  Leaf, 
  BarChart3, 
  Scan, 
  GitCompare, 
  History, 
  Database, 
  Info,
  Cpu
} from 'lucide-react'
import { fetchHealth } from '../services/api'

export default function Navbar() {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline' }))
  }, [])

  return (
    <header className="navbar">
      <div className="nav-inner">
        <NavLink to="/" className="nav-brand">
          <div style={{
            background: 'linear-gradient(135deg, #15803d, #22c55e)',
            color: 'white',
            padding: '0.45rem',
            borderRadius: '10px',
            display: 'flex',
            alignItems: 'center'
          }}>
            <Leaf size={20} />
          </div>
          <div>
            <span>CropGuard AI</span>
            <span className="brand-badge" style={{ marginLeft: '0.5rem' }}>PlantSeg v2</span>
          </div>
        </NavLink>

        <nav className="nav-links">
          <NavLink to="/" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`} end>
            Overview
          </NavLink>
          <NavLink to="/dashboard" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <BarChart3 size={16} />
            Dashboard
          </NavLink>
          <NavLink to="/analyze" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <Scan size={16} />
            Analyze Image
          </NavLink>
          <NavLink to="/models" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <GitCompare size={16} />
            Model Benchmark
          </NavLink>
          <NavLink to="/history" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <History size={16} />
            History
          </NavLink>
          <NavLink to="/dataset" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <Database size={16} />
            Dataset Explorer
          </NavLink>
          <NavLink to="/about" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <Info size={16} />
            Research Info
          </NavLink>
        </nav>

        <div className="nav-status">
          <div 
            className="status-dot" 
            style={{ background: health?.status === 'healthy' ? '#22c55e' : '#ef4444' }} 
          />
          <Cpu size={14} style={{ color: 'var(--text-muted)' }} />
          <span>{health?.device?.toUpperCase() || 'CPU'}</span>
        </div>
      </div>
    </header>
  )
}
