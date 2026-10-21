import React from 'react'
import { ShieldCheck, Heart } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ShieldCheck size={18} style={{ color: 'var(--color-primary)' }} />
          <span>Major Project: <strong>Food Security Through Intelligent Crop Health Monitoring</strong></span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <span>Dataset: <strong>PlantSeg (11,458 images)</strong></span>
          <span>Architectures: <strong>DeepLabV3+ | U-Net | SegFormer-B0</strong></span>
          <span>© 2026 Academic Deep Learning Research</span>
        </div>
      </div>
    </footer>
  )
}
