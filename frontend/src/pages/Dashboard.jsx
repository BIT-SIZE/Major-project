import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { 
  BarChart3, 
  Activity, 
  Scan, 
  Clock, 
  Layers, 
  ArrowRight, 
  TrendingUp,
  Image as ImageIcon,
  Bug,
  Sprout
} from 'lucide-react'
import { fetchAnalyticsSummary, fetchPredictionHistory, fetchHealth } from '../services/api'

export default function Dashboard() {
  const [analytics, setAnalytics] = useState(null)
  const [recentHistory, setRecentHistory] = useState([])
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fetchAnalyticsSummary(),
      fetchPredictionHistory(1, 6),
      fetchHealth()
    ])
      .then(([aData, hData, hlth]) => {
        setAnalytics(aData)
        setRecentHistory(hData.items || [])
        setHealth(hlth)
      })
      .catch((err) => console.error('Dashboard load error:', err))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '5rem 0' }}>
        <Activity className="animate-spin" size={32} style={{ color: 'var(--color-primary)' }} />
        <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>Loading analytics dashboard...</p>
      </div>
    )
  }

  const sevDist = analytics?.severity_distribution || {}
  const totalPreds = analytics?.total_predictions || 0
  const topDiseases = analytics?.top_diagnosed_diseases || {}
  const topHosts = analytics?.top_diagnosed_hosts || {}

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Top Banner */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Crop Health Monitoring Dashboard</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            Empirical monitoring metrics derived directly from connected SQLite database.
          </p>
        </div>
        <Link to="/analyze" className="btn btn-primary">
          <Scan size={16} />
          Analyze New Image
        </Link>
      </div>

      {/* KPI Cards */}
      <div className="grid-4">
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Total Inferences</span>
            <ImageIcon size={18} style={{ color: 'var(--color-primary)' }} />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--text-main)' }}>
            {totalPreds}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            {totalPreds === 0 ? 'No analyses run yet' : 'Logged in database'}
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Avg Severity Damage</span>
            <Activity size={18} style={{ color: '#eab308' }} />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--text-main)' }}>
            {analytics?.average_severity_percentage ? `${analytics.average_severity_percentage}%` : 'N/A'}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Relative to valid leaf tissue area
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Active Classifier</span>
            <Bug size={18} style={{ color: '#0284c7' }} />
          </div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--color-primary-dark)', paddingTop: '0.35rem' }}>
            EFFICIENTNET-B0
          </div>
          <div style={{ fontSize: '0.8rem', color: '#16a34a', fontWeight: 600, marginTop: '0.25rem' }}>
            114 Disease Categories
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Active Segmentation</span>
            <Layers size={18} style={{ color: '#8b5cf6' }} />
          </div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--color-primary-dark)', paddingTop: '0.35rem' }}>
            {health?.selected_segmentation_model?.toUpperCase() || 'DEEPLABV3+'}
          </div>
          <div style={{ fontSize: '0.8rem', color: '#16a34a', fontWeight: 600, marginTop: '0.25rem' }}>
            ASPP ResNet-50 Active
          </div>
        </div>
      </div>

      {/* Disease Pathology & Severity Distribution */}
      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">
              <Bug size={18} style={{ color: 'var(--color-primary)' }} />
              Most Frequently Diagnosed Pathologies
            </h2>
          </div>

          {Object.keys(topDiseases).length === 0 ? (
            <div style={{ padding: '1.5rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              No disease classifications recorded yet. Run an analysis to populate diagnosis statistics.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {Object.entries(topDiseases).map(([disease, count]) => (
                <div key={disease} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.5rem 0.75rem', background: 'var(--bg-subtle)', borderRadius: '6px' }}>
                  <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>{disease}</span>
                  <span style={{ fontWeight: 700, color: 'var(--color-primary-dark)' }}>{count} diagnoses</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">
              <TrendingUp size={18} style={{ color: 'var(--color-primary)' }} />
              Disease Severity Distribution
            </h2>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {[
              { label: 'Healthy (0%)', key: 'Healthy', color: '#16a34a' },
              { label: 'Mild (>0% to 10%)', key: 'Mild', color: '#ca8a04' },
              { label: 'Moderate (>10% to 25%)', key: 'Moderate', color: '#ea580c' },
              { label: 'Severe (>25%)', key: 'Severe', color: '#dc2626' }
            ].map(tier => {
              const count = sevDist[tier.key] || 0
              const pct = totalPreds > 0 ? ((count / totalPreds) * 100).toFixed(0) : 0
              return (
                <div key={tier.key}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                    <span style={{ fontWeight: 600 }}>{tier.label}</span>
                    <span style={{ color: 'var(--text-muted)' }}>{count} ({pct}%)</span>
                  </div>
                  <div style={{ height: '8px', background: 'var(--bg-subtle)', borderRadius: '9999px', overflow: 'hidden' }}>
                    <div style={{ width: `${pct}%`, height: '100%', background: tier.color, borderRadius: '9999px' }} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Recent Predictions Table */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            <Scan size={18} style={{ color: 'var(--color-primary)' }} />
            Recent Plant Inferences & Diagnoses
          </h2>
          <Link to="/history" style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--color-primary)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            View Complete History <ArrowRight size={14} />
          </Link>
        </div>

        {recentHistory.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '2.5rem 1rem' }}>
            <ImageIcon size={40} style={{ color: 'var(--text-subtle)', margin: '0 auto 0.75rem' }} />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 600 }}>No predictions logged yet</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', maxWidth: '400px', margin: '0.25rem auto 1.25rem' }}>
              Upload any plant foliage image to generate pixel-level lesion segmentation and severity estimation.
            </p>
            <Link to="/analyze" className="btn btn-primary btn-sm">
              Launch Image Analysis
            </Link>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-light)', textAlign: 'left', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '0.75rem 0.5rem' }}>Preview</th>
                  <th style={{ padding: '0.75rem 0.5rem' }}>Date & Time</th>
                  <th style={{ padding: '0.75rem 0.5rem' }}>Predicted Pathology</th>
                  <th style={{ padding: '0.75rem 0.5rem' }}>Host Plant</th>
                  <th style={{ padding: '0.75rem 0.5rem' }}>Severity (%)</th>
                  <th style={{ padding: '0.75rem 0.5rem' }}>Category</th>
                  <th style={{ padding: '0.75rem 0.5rem' }}>Health Score</th>
                </tr>
              </thead>
              <tbody>
                {recentHistory.map(item => (
                  <tr key={item.id} style={{ borderBottom: '1px solid var(--bg-subtle)' }}>
                    <td style={{ padding: '0.65rem 0.5rem' }}>
                      <img 
                        src={item.overlay_image_url || item.original_image_url} 
                        alt="crop thumbnail"
                        style={{ width: '44px', height: '44px', objectFit: 'cover', borderRadius: '6px', border: '1px solid var(--border-light)' }} 
                      />
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem', color: 'var(--text-muted)' }}>{item.timestamp}</td>
                    <td style={{ padding: '0.65rem 0.5rem', fontWeight: 700, color: 'var(--text-main)' }}>
                      {item.predicted_disease || 'Unclassified'}
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem' }}>
                      {item.predicted_host ? (
                        <span style={{ background: 'var(--bg-subtle)', padding: '0.2rem 0.5rem', borderRadius: '4px', fontSize: '0.8rem', fontWeight: 600 }}>
                          {item.predicted_host}
                        </span>
                      ) : 'Unknown'}
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem', fontWeight: 700 }}>
                      {item.severity_percentage !== null ? `${item.severity_percentage}%` : 'N/A'}
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem' }}>
                      {item.severity_category ? (
                        <span className={`badge badge-${item.severity_category.toLowerCase()}`}>
                          {item.severity_category}
                        </span>
                      ) : (
                        <span style={{ color: 'var(--text-subtle)' }}>Unavailable</span>
                      )}
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem', fontWeight: 600, color: 'var(--color-primary-dark)' }}>
                      {item.health_indicator !== null ? `${item.health_indicator} / 100` : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
