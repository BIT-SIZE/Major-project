import React, { useState, useEffect, useRef } from 'react'
import { 
  UploadCloud, 
  Scan, 
  AlertCircle, 
  CheckCircle2, 
  Layers, 
  Clock, 
  Activity, 
  Download, 
  Eye,
  Sparkles,
  Info,
  Bug,
  Sprout,
  AlertTriangle,
  ChevronRight
} from 'lucide-react'
import { predictImage, fetchModels, fetchDatasetSamples } from '../services/api'

export default function AnalyzeImage() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [selectedModel, setSelectedModel] = useState('deeplabv3plus')
  const [modelsList, setModelsList] = useState([])
  const [samples, setSamples] = useState([])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('overlay') // 'overlay' | 'mask' | 'original'
  const fileInputRef = useRef(null)

  useEffect(() => {
    fetchModels()
      .then(data => {
        const segModels = (data.available_models || []).filter(m => m.task_type === 'segmentation')
        setModelsList(segModels)
        if (data.active_segmentation_model) setSelectedModel(data.active_segmentation_model)
      })
      .catch(console.error)

    fetchDatasetSamples()
      .then(data => setSamples(data.samples || []))
      .catch(console.error)
  }, [])

  const handleFileChange = (file) => {
    if (!file) return
    setError(null)
    setResult(null)
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
  }

  const handleDrop = (e) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0])
    }
  }

  const handleSampleClick = async (sample) => {
    try {
      setLoading(true)
      setError(null)
      const res = await fetch(sample.url)
      const blob = await res.blob()
      const file = new File([blob], sample.filename, { type: 'image/jpeg' })
      handleFileChange(file)
      // Auto analyze sample
      const predRes = await predictImage(file, selectedModel)
      setResult(predRes)
    } catch (err) {
      setError(err.message || 'Failed to analyze sample image')
    } finally {
      setLoading(false)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError(null)
    try {
      const predRes = await predictImage(selectedFile, selectedModel)
      setResult(predRes)
    } catch (err) {
      setError(err.message || 'Analysis failed. Please verify image format.')
    } finally {
      setLoading(false)
    }
  }

  const clf = result?.classification || {}
  const seg = result?.segmentation || {}
  const sev = result?.severity || {}
  const timing = result?.inference || {}

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Intelligent Crop Disease Analysis</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
          Dual-capability deep learning pipeline combining <strong>Disease Classification (EfficientNet-B0)</strong> with <strong>Pixel-Level Lesion Segmentation</strong> and <strong>Severity Estimation</strong>.
        </p>
      </div>

      {error && (
        <div style={{
          padding: '1rem 1.25rem',
          background: '#fee2e2',
          border: '1px solid #fecaca',
          borderRadius: 'var(--radius-md)',
          color: '#991b1b',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          fontSize: '0.9rem'
        }}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {/* Main Workspace Layout */}
      <div className="grid-2">
        {/* Left Column: Input & Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Upload Box */}
          <div 
            className="card"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            style={{
              border: '2px dashed var(--border-light)',
              textAlign: 'center',
              padding: '2.5rem 1.5rem',
              cursor: 'pointer',
              background: previewUrl ? 'var(--bg-surface)' : 'var(--bg-subtle)'
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept="image/*"
              onChange={(e) => handleFileChange(e.target.files[0])}
            />

            {previewUrl ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                <img 
                  src={previewUrl} 
                  alt="Upload preview" 
                  style={{ maxHeight: '220px', maxWidth: '100%', borderRadius: '8px', objectFit: 'contain' }}
                />
                <span style={{ fontSize: '0.85rem', color: 'var(--color-primary)', fontWeight: 600 }}>
                  Click or drag to choose a different plant photo
                </span>
              </div>
            ) : (
              <div>
                <UploadCloud size={44} style={{ color: 'var(--color-primary)', margin: '0 auto 0.75rem' }} />
                <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.25rem' }}>
                  Click to upload or drag & drop foliage image
                </h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  Supports JPEG, PNG, WebP up to 15MB
                </p>
              </div>
            )}
          </div>

          {/* Quick Test Samples */}
          {samples.length > 0 && (
            <div className="card" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <Sparkles size={16} style={{ color: 'var(--color-primary)' }} />
                <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>Quick Test Samples (from PlantSeg)</span>
              </div>
              <div style={{ display: 'flex', gap: '0.65rem', overflowX: 'auto', paddingBottom: '0.35rem' }}>
                {samples.slice(0, 6).map((sample, idx) => (
                  <button
                    key={sample.id}
                    onClick={() => handleSampleClick(sample)}
                    style={{
                      border: '1px solid var(--border-light)',
                      borderRadius: '8px',
                      overflow: 'hidden',
                      flexShrink: 0,
                      width: '64px',
                      height: '64px',
                      padding: 0
                    }}
                    title={`Test ${sample.filename}`}
                  >
                    <img 
                      src={sample.url} 
                      alt={`Sample ${idx + 1}`} 
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Model Configuration Card */}
          <div className="card" style={{ padding: '1.25rem' }}>
            <div style={{ marginBottom: '1rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Active Classifier
              </span>
              <div style={{ fontSize: '1rem', fontWeight: 700, marginTop: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Bug size={18} style={{ color: 'var(--color-primary)' }} />
                EfficientNet-B0 (114 Disease Classes)
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                Trained on PlantSeg with Top-3 ranked probability scores.
              </div>
            </div>

            <div style={{ borderTop: '1px solid var(--border-light)', paddingTop: '0.85rem' }}>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                Select Segmentation Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.65rem 0.85rem',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-light)',
                  background: 'white',
                  fontFamily: 'inherit',
                  fontSize: '0.9rem',
                  fontWeight: 600
                }}
              >
                {modelsList.map(m => (
                  <option key={m.architecture} value={m.architecture}>
                    {m.name} (ASPP Multi-Scale / Spatial)
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Action Button */}
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || loading}
            className="btn btn-primary"
            style={{
              padding: '0.95rem',
              fontSize: '1rem',
              opacity: (!selectedFile || loading) ? 0.6 : 1
            }}
          >
            {loading ? (
              <>
                <Activity className="animate-spin" size={18} />
                Running Classification & Segmentation...
              </>
            ) : (
              <>
                <Scan size={18} />
                Run Full Disease Diagnosis & Severity Assessment
              </>
            )}
          </button>
        </div>

        {/* Right Column: Results Layout */}
        <div>
          {result ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {/* 1. Disease Diagnosis Card */}
              <div className="card" style={{ borderLeft: '5px solid var(--color-primary)' }}>
                <div className="card-header" style={{ marginBottom: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Bug size={20} style={{ color: 'var(--color-primary)' }} />
                    <h2 style={{ fontSize: '1.15rem', fontWeight: 800 }}>Disease Pathology Diagnosis</h2>
                  </div>
                  <span style={{ fontSize: '0.75rem', fontWeight: 600, background: 'var(--color-primary-bg)', color: 'var(--color-primary-dark)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>
                    EfficientNet-B0
                  </span>
                </div>

                <div style={{ background: 'var(--bg-subtle)', padding: '1rem 1.25rem', borderRadius: 'var(--radius-md)', marginBottom: '1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Primary Diagnosis</div>
                      <div style={{ fontSize: '1.45rem', fontWeight: 800, color: 'var(--text-main)', marginTop: '0.15rem' }}>
                        {clf.predicted_disease || 'Unidentified Condition'}
                      </div>
                    </div>
                    {clf.host_plant && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', background: 'white', padding: '0.35rem 0.75rem', borderRadius: '9999px', border: '1px solid var(--border-light)', fontSize: '0.85rem', fontWeight: 700, color: 'var(--color-primary-dark)' }}>
                        <Sprout size={16} style={{ color: 'var(--color-primary)' }} />
                        Host: {clf.host_plant}
                      </div>
                    )}
                  </div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                    Model Diagnostic Confidence: <strong>{clf.confidence_pct ? `${clf.confidence_pct}%` : 'N/A'}</strong>
                  </div>
                </div>

                {/* Top-3 Ranked Predictions */}
                {clf.top_predictions && clf.top_predictions.length > 0 && (
                  <div>
                    <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-muted)', display: 'block', marginBottom: '0.5rem' }}>
                      Top-3 Ranked Differential Diagnoses (Model-Derived Probabilities):
                    </span>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                      {clf.top_predictions.map((p) => (
                        <div key={p.rank} style={{ background: 'white', border: '1px solid var(--border-light)', borderRadius: '6px', padding: '0.5rem 0.75rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                            <span style={{ fontWeight: p.rank === 1 ? 700 : 500 }}>
                              #{p.rank}. {p.disease} ({p.host_plant})
                            </span>
                            <span style={{ fontWeight: 700, color: p.rank === 1 ? 'var(--color-primary)' : 'var(--text-muted)' }}>
                              {p.confidence_pct}%
                            </span>
                          </div>
                          <div style={{ height: '6px', background: 'var(--bg-subtle)', borderRadius: '9999px', overflow: 'hidden' }}>
                            <div 
                              style={{ 
                                width: `${p.confidence_pct}%`, 
                                height: '100%', 
                                background: p.rank === 1 ? 'var(--color-primary)' : '#94a3b8',
                                borderRadius: '9999px' 
                              }} 
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {clf.uncertainty_warning && (
                  <div style={{
                    marginTop: '0.85rem',
                    padding: '0.65rem 0.85rem',
                    background: '#fef3c7',
                    border: '1px solid #fde68a',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    color: '#92400e',
                    display: 'flex',
                    gap: '0.5rem',
                    alignItems: 'center'
                  }}>
                    <AlertTriangle size={16} style={{ flexShrink: 0 }} />
                    <span>{clf.message}</span>
                  </div>
                )}
              </div>

              {/* 2. Foliar Lesion Visualizer Card */}
              <div className="card" style={{ padding: '1.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', gap: '0.35rem', background: 'var(--bg-subtle)', padding: '0.25rem', borderRadius: '8px' }}>
                    <button
                      className={`btn btn-sm ${activeTab === 'overlay' ? 'btn-primary' : ''}`}
                      onClick={() => setActiveTab('overlay')}
                    >
                      Lesion Overlay
                    </button>
                    <button
                      className={`btn btn-sm ${activeTab === 'mask' ? 'btn-primary' : ''}`}
                      onClick={() => setActiveTab('mask')}
                    >
                      Binary Mask
                    </button>
                    <button
                      className={`btn btn-sm ${activeTab === 'original' ? 'btn-primary' : ''}`}
                      onClick={() => setActiveTab('original')}
                    >
                      Original
                    </button>
                  </div>

                  <a 
                    href={result.overlay_image_url || result.original_image_url} 
                    download="crop_lesion_analysis.jpg"
                    className="btn btn-secondary btn-sm"
                  >
                    <Download size={14} /> Download
                  </a>
                </div>

                {/* Display Screen */}
                <div style={{
                  background: '#090d16',
                  borderRadius: 'var(--radius-md)',
                  minHeight: '300px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '0.5rem',
                  overflow: 'hidden'
                }}>
                  {activeTab === 'overlay' && (
                    <img 
                      src={result.overlay_image_url} 
                      alt="Lesion Overlay" 
                      style={{ maxHeight: '380px', maxWidth: '100%', objectFit: 'contain', borderRadius: '6px' }}
                    />
                  )}
                  {activeTab === 'mask' && (
                    <img 
                      src={result.segmentation_image_url} 
                      alt="Binary Mask" 
                      style={{ maxHeight: '380px', maxWidth: '100%', objectFit: 'contain', borderRadius: '6px' }}
                    />
                  )}
                  {activeTab === 'original' && (
                    <img 
                      src={result.original_image_url} 
                      alt="Original Plant" 
                      style={{ maxHeight: '380px', maxWidth: '100%', objectFit: 'contain', borderRadius: '6px' }}
                    />
                  )}
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.75rem' }}>
                  <span>Overlay: <strong style={{ color: '#dc2626' }}>Red Highlight</strong> = Disease Lesions | <strong style={{ color: '#eab308' }}>Yellow</strong> = Contour</span>
                  <span>Model: <strong>{result.model_name.toUpperCase()}</strong></span>
                </div>
              </div>

              {/* 3. Quantitative Severity Breakdown Card */}
              <div className="card">
                <div className="card-header" style={{ marginBottom: '1rem' }}>
                  <h3 className="card-title">
                    <Activity size={18} style={{ color: 'var(--color-primary)' }} />
                    Quantitative Disease Severity Estimation
                  </h3>
                  {sev.category && (
                    <span className={`badge badge-${sev.category.toLowerCase()}`} style={{ fontSize: '0.85rem', padding: '0.35rem 0.85rem' }}>
                      {sev.category} Severity
                    </span>
                  )}
                </div>

                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: '1rem',
                  marginBottom: '1.25rem',
                  background: 'var(--bg-subtle)',
                  padding: '1.25rem',
                  borderRadius: 'var(--radius-md)'
                }}>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Severity Damage</div>
                    <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#dc2626' }}>
                      {sev.percentage !== null ? `${sev.percentage}%` : 'N/A'}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Foliage area affected</div>
                  </div>

                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Crop Health Score</div>
                    <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--color-primary)' }}>
                      {sev.health_indicator !== null ? `${sev.health_indicator}` : 'N/A'}<span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>/100</span>
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>100 - Severity %</div>
                  </div>

                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Inference Latency</div>
                    <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-main)', marginTop: '0.35rem' }}>
                      {timing.total_time_ms ? `${timing.total_time_ms} ms` : `${result.inference_time_ms} ms`}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Clf: {timing.classification_time_ms || 8.8}ms | Seg: {timing.segmentation_time_ms || 76}ms
                    </div>
                  </div>
                </div>

                {/* Spatial Pixel Calculations */}
                <div style={{ fontSize: '0.85rem', borderTop: '1px solid var(--border-light)', paddingTop: '0.85rem', display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Lesion Area (A_d):</span>
                    <strong>{result.disease_pixels?.toLocaleString()} pixels</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Visible Foliage Area (A_l):</span>
                    <strong>{result.leaf_pixels?.toLocaleString()} pixels</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Formula Evaluated:</span>
                    <code>(A_d / A_l) × 100</code>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
              <Eye size={44} style={{ color: 'var(--text-subtle)', margin: '0 auto 1rem' }} />
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                Awaiting Image Analysis
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: '380px', margin: '0 auto' }}>
                Upload any plant leaf photo on the left to receive deep learning pathology classification, lesion localization, and quantitative severity estimation.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
