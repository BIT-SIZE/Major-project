import React from 'react'
import { Link } from 'react-router-dom'
import { 
  Scan, 
  BarChart3, 
  GitCompare, 
  ShieldAlert, 
  Layers, 
  Activity, 
  ArrowRight,
  CheckCircle2,
  Cpu
} from 'lucide-react'

export default function LandingPage() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3.5rem' }}>
      {/* Hero Section */}
      <section style={{
        background: 'linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%)',
        borderRadius: 'var(--radius-xl)',
        padding: '3.5rem 2.5rem',
        border: '1px solid var(--color-primary-border)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{ maxWidth: '800px' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            background: 'white',
            padding: '0.35rem 0.85rem',
            borderRadius: '9999px',
            fontSize: '0.8rem',
            fontWeight: 700,
            color: 'var(--color-primary-dark)',
            border: '1px solid var(--color-primary-border)',
            marginBottom: '1.25rem',
            boxShadow: 'var(--shadow-sm)'
          }}>
            <Activity size={14} />
            Deep Learning Computer Vision System
          </div>

          <h1 style={{
            fontSize: '2.6rem',
            fontWeight: 800,
            lineHeight: 1.2,
            color: '#0f172a',
            marginBottom: '1rem'
          }}>
            Food Security Through Intelligent Crop Health Monitoring
          </h1>

          <p style={{
            fontSize: '1.1rem',
            color: 'var(--text-muted)',
            lineHeight: 1.6,
            marginBottom: '2rem'
          }}>
            Moving beyond simple binary classification to <strong>pixel-level semantic segmentation</strong> and <strong>quantitative disease severity estimation</strong>. Accurately localize plant disease lesions and measure damaged foliage area to safeguard agricultural productivity.
          </p>

          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <Link to="/analyze" className="btn btn-primary" style={{ padding: '0.85rem 1.75rem', fontSize: '1rem' }}>
              <Scan size={18} />
              Start Image Analysis
              <ArrowRight size={16} />
            </Link>
            <Link to="/models" className="btn btn-secondary" style={{ padding: '0.85rem 1.5rem', fontSize: '1rem' }}>
              <GitCompare size={18} />
              View Benchmark Experiments
            </Link>
          </div>
        </div>
      </section>

      {/* Problem vs Solution */}
      <section className="grid-2">
        <div className="card" style={{ borderLeft: '4px solid #ef4444' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#dc2626', marginBottom: '0.75rem' }}>
            <ShieldAlert size={20} />
            <h2 style={{ fontSize: '1.2rem', fontWeight: 700 }}>Conventional Limitations</h2>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', lineHeight: 1.6 }}>
            Most conventional agricultural machine learning applications perform only image-level classification (labeling an image simply as "healthy" or "diseased"). This provides no information on where disease lesions are situated, gives zero indication of how much plant tissue is infected, and cannot quantify disease severity.
          </p>
        </div>

        <div className="card" style={{ borderLeft: '4px solid var(--color-primary)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-primary)', marginBottom: '0.75rem' }}>
            <CheckCircle2 size={20} />
            <h2 style={{ fontSize: '1.2rem', fontWeight: 700 }}>Our Quantitative Solution</h2>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', lineHeight: 1.6 }}>
            We implement semantic segmentation architectures (DeepLabV3+, U-Net, and SegFormer) trained on the real-world PlantSeg dataset. The system predicts exact lesion masks and calculates the quantitative disease severity ratio strictly relative to valid leaf tissue area: <code>(A_lesion / A_leaf) × 100</code>.
          </p>
        </div>
      </section>

      {/* Core Capabilities */}
      <section>
        <div style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 800 }}>Core Engineering Capabilities</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>Full-stack integration from raw image processing to deep learning inference and monitoring analytics.</p>
        </div>

        <div className="grid-3">
          <div className="card">
            <div style={{
              width: '42px',
              height: '42px',
              borderRadius: '10px',
              background: '#dcfce7',
              color: '#15803d',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '1rem'
            }}>
              <Scan size={22} />
            </div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.5rem' }}>Pixel-Level Lesion Segmentation</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Identifies irregular boundaries of foliar lesions in complex natural field environments with variable lighting and backgrounds.
            </p>
          </div>

          <div className="card">
            <div style={{
              width: '42px',
              height: '42px',
              borderRadius: '10px',
              background: '#e0f2fe',
              color: '#0284c7',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '1rem'
            }}>
              <Layers size={22} />
            </div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.5rem' }}>Quantitative Severity Ratio</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Calculates percentage damage relative strictly to visible foliage area rather than image borders, categorizing into Healthy, Mild, Moderate, or Severe.
            </p>
          </div>

          <div className="card">
            <div style={{
              width: '42px',
              height: '42px',
              borderRadius: '10px',
              background: '#fef3c7',
              color: '#b45309',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '1rem'
            }}>
              <GitCompare size={22} />
            </div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.5rem' }}>Empirical Architecture Comparison</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Evaluates DeepLabV3+ (ResNet-50), U-Net, and SegFormer-B0 on identical PlantSeg test splits across IoU, Dice, Precision, Recall, and inference latency.
            </p>
          </div>
        </div>
      </section>

      {/* Workflow Section */}
      <section className="card" style={{ padding: '2.5rem' }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 800, marginBottom: '1.5rem', textAlign: 'center' }}>
          Real-Time Pipeline Architecture
        </h2>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(5, 1fr)',
          gap: '1rem',
          textAlign: 'center'
        }}>
          <div style={{ padding: '1rem', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-md)' }}>
            <div style={{ fontWeight: 700, color: 'var(--color-primary)', marginBottom: '0.25rem' }}>1. Upload</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Plant Image Ingestion & RGB Validation</div>
          </div>
          <div style={{ padding: '1rem', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-md)' }}>
            <div style={{ fontWeight: 700, color: 'var(--color-primary)', marginBottom: '0.25rem' }}>2. Foliage Mask</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Adaptive Leaf Boundary Extraction</div>
          </div>
          <div style={{ padding: '1rem', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-md)' }}>
            <div style={{ fontWeight: 700, color: 'var(--color-primary)', marginBottom: '0.25rem' }}>3. Model Inference</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>DeepLabV3+ / U-Net / SegFormer Segmentation</div>
          </div>
          <div style={{ padding: '1rem', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-md)' }}>
            <div style={{ fontWeight: 700, color: 'var(--color-primary)', marginBottom: '0.25rem' }}>4. Severity Math</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Lesion Area vs Leaf Area Ratio Calculation</div>
          </div>
          <div style={{ padding: '1rem', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-md)' }}>
            <div style={{ fontWeight: 700, color: 'var(--color-primary)', marginBottom: '0.25rem' }}>5. Dashboard</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Interactive Overlay & History Persistence</div>
          </div>
        </div>
      </section>
    </div>
  )
}
