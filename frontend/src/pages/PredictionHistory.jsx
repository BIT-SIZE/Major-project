import React, { useState, useEffect } from 'react'
import { History, Trash2, Filter, Eye, X, AlertCircle } from 'lucide-react'
import { fetchPredictionHistory, deletePrediction } from '../services/api'

export default function PredictionHistory() {
  const [historyData, setHistoryData] = useState({ total: 0, items: [] })
  const [loading, setLoading] = useState(true)
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('')
  const [selectedItem, setSelectedItem] = useState(null)
  const [page, setPage] = useState(1)

  const loadHistory = () => {
    setLoading(true)
    fetchPredictionHistory(page, 15, selectedModel || null, selectedCategory || null)
      .then(setHistoryData)
      .catch(console.error)
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    loadHistory()
  }, [page, selectedModel, selectedCategory])

  const handleDelete = async (id, e) => {
    e.stopPropagation()
    if (!window.confirm('Are you sure you want to delete this prediction record?')) return
    try {
      await deletePrediction(id)
      loadHistory()
      if (selectedItem?.id === id) setSelectedItem(null)
    } catch (err) {
      alert('Failed to delete record: ' + err.message)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Inference Prediction History</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            Historical analysis logs and segmentations stored in local SQLite database.
          </p>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="card" style={{ padding: '1rem 1.25rem' }}>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.85rem', fontWeight: 600 }}>
            <Filter size={16} /> Filters:
          </div>

          <select
            value={selectedModel}
            onChange={(e) => { setSelectedModel(e.target.value); setPage(1); }}
            style={{
              padding: '0.45rem 0.75rem',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-light)',
              background: 'white',
              fontSize: '0.85rem'
            }}
          >
            <option value="">All Models</option>
            <option value="deeplabv3plus">DeepLabV3+</option>
            <option value="unet">U-Net</option>
            <option value="segformer">SegFormer-B0</option>
          </select>

          <select
            value={selectedCategory}
            onChange={(e) => { setSelectedCategory(e.target.value); setPage(1); }}
            style={{
              padding: '0.45rem 0.75rem',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-light)',
              background: 'white',
              fontSize: '0.85rem'
            }}
          >
            <option value="">All Severity Tiers</option>
            <option value="Healthy">Healthy (0%)</option>
            <option value="Mild">Mild (0-10%)</option>
            <option value="Moderate">Moderate (10-25%)</option>
            <option value="Severe">Severe (&gt;25%)</option>
          </select>

          {(selectedModel || selectedCategory) && (
            <button
              onClick={() => { setSelectedModel(''); setSelectedCategory(''); setPage(1); }}
              style={{ fontSize: '0.8rem', color: '#dc2626', fontWeight: 600 }}
            >
              Reset Filters
            </button>
          )}

          <div style={{ marginLeft: 'auto', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Total: <strong>{historyData.total}</strong> records
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="card">
        {loading ? (
          <div style={{ textAlign: 'center', padding: '3rem' }}>
            <p style={{ color: 'var(--text-muted)' }}>Loading history records...</p>
          </div>
        ) : historyData.items.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '3rem 1rem' }}>
            <History size={40} style={{ color: 'var(--text-subtle)', margin: '0 auto 0.75rem' }} />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>No prediction records found</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.25rem' }}>
              Run image analyses to populate your historical monitoring database.
            </p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-light)', textAlign: 'left', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '0.75rem 0.75rem' }}>Thumb</th>
                  <th style={{ padding: '0.75rem 0.75rem' }}>Timestamp</th>
                  <th style={{ padding: '0.75rem 0.75rem' }}>Predicted Pathology</th>
                  <th style={{ padding: '0.75rem 0.75rem' }}>Host Plant</th>
                  <th style={{ padding: '0.75rem 0.75rem' }}>Confidence</th>
                  <th style={{ padding: '0.75rem 0.75rem' }}>Severity (%)</th>
                  <th style={{ padding: '0.75rem 0.75rem' }}>Category</th>
                  <th style={{ padding: '0.75rem 0.75rem' }}>Health Score</th>
                  <th style={{ padding: '0.75rem 0.75rem' }}>Model</th>
                  <th style={{ padding: '0.75rem 0.75rem', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {historyData.items.map((r) => (
                  <tr 
                    key={r.id} 
                    style={{ borderBottom: '1px solid var(--bg-subtle)', cursor: 'pointer' }}
                    onClick={() => setSelectedItem(r)}
                  >
                    <td style={{ padding: '0.65rem 0.75rem' }}>
                      <img 
                        src={r.overlay_image_url || r.original_image_url} 
                        alt="thumb" 
                        style={{ width: '48px', height: '48px', objectFit: 'cover', borderRadius: '6px', border: '1px solid var(--border-light)' }} 
                      />
                    </td>
                    <td style={{ padding: '0.65rem 0.75rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>{r.timestamp}</td>
                    <td style={{ padding: '0.65rem 0.75rem', fontWeight: 600, color: 'var(--text-main)' }}>
                      {r.predicted_disease ? (
                        <span style={{ textTransform: 'capitalize' }}>{r.predicted_disease}</span>
                      ) : (
                        <span style={{ color: 'var(--text-subtle)' }}>Unclassified</span>
                      )}
                    </td>
                    <td style={{ padding: '0.65rem 0.75rem' }}>
                      {r.predicted_host ? (
                        <span className="badge" style={{ background: '#f0fdf4', color: '#166534', border: '1px solid #bbf7d0', fontSize: '0.75rem', textTransform: 'capitalize' }}>
                          {r.predicted_host}
                        </span>
                      ) : (
                        <span style={{ color: 'var(--text-subtle)' }}>-</span>
                      )}
                    </td>
                    <td style={{ padding: '0.65rem 0.75rem', fontWeight: 600, color: 'var(--color-primary-dark)' }}>
                      {r.classification_score ? `${(r.classification_score * 100).toFixed(1)}%` : '-'}
                    </td>
                    <td style={{ padding: '0.65rem 0.75rem', fontWeight: 700 }}>
                      {r.severity_percentage !== null ? `${r.severity_percentage}%` : 'N/A'}
                    </td>
                    <td style={{ padding: '0.65rem 0.75rem' }}>
                      {r.severity_category ? (
                        <span className={`badge badge-${r.severity_category.toLowerCase()}`}>
                          {r.severity_category}
                        </span>
                      ) : (
                        <span style={{ color: 'var(--text-subtle)' }}>N/A</span>
                      )}
                    </td>
                    <td style={{ padding: '0.65rem 0.75rem', fontWeight: 600, color: 'var(--color-primary-dark)' }}>
                      {r.health_indicator !== null ? `${r.health_indicator} / 100` : 'N/A'}
                    </td>
                    <td style={{ padding: '0.65rem 0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      {r.model_name.toUpperCase()}
                    </td>
                    <td style={{ padding: '0.65rem 0.75rem', textAlign: 'right' }}>
                      <button
                        onClick={(e) => handleDelete(r.id, e)}
                        style={{ color: '#ef4444', padding: '0.35rem', borderRadius: '4px' }}
                        title="Delete Record"
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Detail Inspection Modal */}
      {selectedItem && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.65)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 100,
          padding: '1rem'
        }}>
          <div className="card" style={{ maxWidth: '780px', width: '100%', maxHeight: '90vh', overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h2 style={{ fontSize: '1.2rem', fontWeight: 800 }}>Prediction Detail: {selectedItem.id}</h2>
              <button onClick={() => setSelectedItem(null)} style={{ padding: '0.25rem' }}>
                <X size={20} />
              </button>
            </div>

            {/* Disease Pathology Diagnosis Banner */}
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '1rem', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div>
                  <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                    Predicted Plant Pathology
                  </span>
                  <div style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--text-main)', textTransform: 'capitalize' }}>
                    {selectedItem.predicted_disease || 'Unclassified Pathology'}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  {selectedItem.predicted_host && (
                    <span className="badge" style={{ background: '#ecfdf5', color: '#065f46', border: '1px solid #a7f3d0', fontSize: '0.85rem', fontWeight: 600, textTransform: 'capitalize' }}>
                      Host: {selectedItem.predicted_host}
                    </span>
                  )}
                  {selectedItem.classification_score && (
                    <span className="badge" style={{ background: '#eff6ff', color: '#1e40af', border: '1px solid #bfdbfe', fontSize: '0.85rem', fontWeight: 700 }}>
                      Score: {(selectedItem.classification_score * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              </div>

              {/* Differential Diagnoses (Top-3) if available */}
              {selectedItem.top_predictions && selectedItem.top_predictions.length > 0 && (
                <div style={{ marginTop: '0.5rem' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                    Differential Diagnoses (Top-{selectedItem.top_predictions.length}):
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                    {selectedItem.top_predictions.map((p, idx) => (
                      <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}>
                        <span style={{ width: '16px', fontWeight: 700, color: 'var(--text-muted)' }}>#{idx + 1}</span>
                        <span style={{ width: '160px', textTransform: 'capitalize', fontWeight: idx === 0 ? 700 : 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {p.disease}
                        </span>
                        <div style={{ flex: 1, height: '6px', background: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${Math.min(100, Math.round(p.score * 100))}%`, height: '100%', background: idx === 0 ? 'var(--color-primary)' : '#94a3b8' }} />
                        </div>
                        <span style={{ width: '45px', textAlign: 'right', fontWeight: 600, color: 'var(--text-muted)' }}>
                          {(p.score * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
              <div>
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-muted)' }}>Lesion Overlay</span>
                <img 
                  src={selectedItem.overlay_image_url} 
                  alt="Overlay" 
                  style={{ width: '100%', borderRadius: '8px', border: '1px solid var(--border-light)', marginTop: '0.25rem' }} 
                />
              </div>
              <div>
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-muted)' }}>Binary Segmentation Mask</span>
                <img 
                  src={selectedItem.mask_image_url} 
                  alt="Mask" 
                  style={{ width: '100%', borderRadius: '8px', border: '1px solid var(--border-light)', marginTop: '0.25rem' }} 
                />
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', background: 'var(--bg-subtle)', padding: '1rem', borderRadius: '8px' }}>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Severity</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#dc2626' }}>
                  {selectedItem.severity_percentage}%
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Category</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
                  {selectedItem.severity_category}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Health Score</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--color-primary)' }}>
                  {selectedItem.health_indicator}/100
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Latency</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
                  {selectedItem.inference_time_ms} ms
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
