import React from 'react'
import { ShieldCheck, BookOpen, Layers, Target, AlertTriangle, ArrowRight, Activity, GitBranch } from 'lucide-react'

export default function AboutProject() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem', maxWidth: '1000px', margin: '0 auto' }}>
      <div>
        <h1 style={{ fontSize: '2.2rem', fontWeight: 800 }}>Project Research & System Architecture</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '1.05rem', marginTop: '0.25rem' }}>
          Major Project: <strong>Food Security Through Intelligent Crop Health Monitoring</strong>
        </p>
      </div>

      {/* Problem Statement Card */}
      <section className="card">
        <h2 style={{ fontSize: '1.3rem', fontWeight: 800, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-primary)' }}>
          <Target size={20} /> Problem Statement & Motivation
        </h2>
        <div style={{ color: 'var(--text-muted)', lineHeight: 1.7, fontSize: '0.95rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <p>
            Global food security is increasingly vulnerable to crop disease outbreaks, changing climate patterns, and intensive farming pressures. Delayed disease diagnosis causes catastrophic yield loss.
          </p>
          <p>
            While computer vision has been applied to crop health, the overwhelming majority of existing systems rely purely on <strong>image-level classification</strong> (predicting labels such as "Corn Common Rust" without localizing lesions). Classification alone cannot establish how much of the leaf tissue is impaired or track disease progression quantitatively.
          </p>
          <p>
            This project bridges this critical gap by delivering <strong>pixel-level semantic segmentation</strong> coupled with <strong>quantitative disease severity assessment</strong>, allowing agronomists and farmers to measure exact damaged tissue percentages.
          </p>
        </div>
      </section>

      {/* Mathematical Severity Formulation */}
      <section className="card">
        <h2 style={{ fontSize: '1.3rem', fontWeight: 800, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-primary)' }}>
          <Activity size={20} /> Mathematical Formulation of Disease Severity
        </h2>
        <div style={{ background: 'var(--bg-subtle)', padding: '1.25rem', borderRadius: 'var(--radius-md)', marginBottom: '1rem' }}>
          <div style={{ fontSize: '1.2rem', fontWeight: 700, fontFamily: 'var(--font-mono)', textAlign: 'center', color: 'var(--color-primary-dark)' }}>
            Disease Severity (%) = (A_d / A_l) × 100
          </div>
        </div>
        <div style={{ color: 'var(--text-muted)', lineHeight: 1.7, fontSize: '0.95rem', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
          <p>
            <strong>A_d:</strong> Number of pixels classified as diseased leaf lesions by the deep learning segmentation network.
          </p>
          <p>
            <strong>A_l:</strong> Number of pixels belonging to visible plant leaf foliage (extracted via adaptive vegetation segmentation).
          </p>
          <p>
            <strong>Why Total Image Area is NOT Used:</strong> In field photography, images contain soil, background clutter, and sky. Dividing lesion pixels by total image dimensions results in artificially deflated severity numbers that fluctuate simply based on camera distance. By constraining the denominator strictly to visible foliage, our metric remains physically meaningful.
          </p>
        </div>
      </section>

      {/* Two-Stage Segmentation Architecture */}
      <section className="card">
        <h2 style={{ fontSize: '1.3rem', fontWeight: 800, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-primary)' }}>
          <GitBranch size={20} /> Two-Stage Experimental Pipeline Design
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', textAlign: 'center', margin: '1rem 0' }}>
          <div style={{ padding: '1rem', background: 'var(--bg-subtle)', borderRadius: '8px' }}>
            <div style={{ fontWeight: 700, fontSize: '0.85rem' }}>Stage 1</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Leaf Canopy Segmentation</div>
          </div>
          <div style={{ padding: '1rem', background: 'var(--bg-subtle)', borderRadius: '8px' }}>
            <div style={{ fontWeight: 700, fontSize: '0.85rem' }}>ROI Masking</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Background Removal</div>
          </div>
          <div style={{ padding: '1rem', background: 'var(--bg-subtle)', borderRadius: '8px' }}>
            <div style={{ fontWeight: 700, fontSize: '0.85rem' }}>Stage 2</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Lesion Semantic Segmentation</div>
          </div>
          <div style={{ padding: '1rem', background: 'var(--bg-subtle)', borderRadius: '8px' }}>
            <div style={{ fontWeight: 700, fontSize: '0.85rem' }}>Output</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Quantitative Ratio & Health Score</div>
          </div>
        </div>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Inspired by two-stage corn disease quantification research. When separate whole-leaf ground truth masks are provided, Stage 1 isolates the entire canopy outline. In the current primary PlantSeg benchmark, Stage 1 uses adaptive foliage extraction and Stage 2 executes DeepLabV3+.
        </p>
      </section>

      {/* Scientific Limitations & Future Extensions */}
      <div className="grid-2">
        <div className="card" style={{ borderLeft: '4px solid #f59e0b' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#b45309' }}>
            <AlertTriangle size={18} /> Current Limitations
          </h3>
          <ul style={{ paddingLeft: '1.25rem', color: 'var(--text-muted)', fontSize: '0.85rem', lineHeight: 1.6, display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <li>Single 2D image views cannot capture the underside of multi-layered canopies.</li>
            <li>PlantSeg lesion masks do not include comprehensive annotated leaf boundaries for every sample.</li>
            <li>Extreme shadow transitions can occasionally register as false positive edges.</li>
          </ul>
        </div>

        <div className="card" style={{ borderLeft: '4px solid var(--color-primary)' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--color-primary)' }}>
            <ShieldCheck size={18} /> Food Security Extensions
          </h3>
          <ul style={{ paddingLeft: '1.25rem', color: 'var(--text-muted)', fontSize: '0.85rem', lineHeight: 1.6, display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <li>Integration with IoT soil moisture and ambient humidity sensors.</li>
            <li>Temporal progression tracking to forecast disease incubation speed.</li>
            <li>Crop yield impact modeling incorporating growth-stage indicators.</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
