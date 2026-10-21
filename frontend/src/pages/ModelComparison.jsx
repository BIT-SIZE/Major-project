import React, { useState, useEffect } from 'react'
import { GitCompare, Award, Zap, Layers, AlertCircle, CheckCircle2 } from 'lucide-react'
import { fetchModelComparison } from '../services/api'

export default function ModelComparison() {
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchModelComparison()
      .then(setReport)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  const results = report?.benchmark_results || []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Comparative Deep Learning Benchmark</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
          Experimental evaluation on official PlantSeg Test split across IoU, Dice, Precision, Recall, Severity MAE, latency, and memory footprint.
        </p>
      </div>

      {/* Top Architecture Highlight Cards */}
      <div className="grid-3">
        <div className="card" style={{ borderTop: '4px solid #16a34a' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#16a34a', marginBottom: '0.5rem' }}>
            <Award size={20} />
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Best Segmentation: DeepLabV3+</h2>
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--text-main)', marginBottom: '0.25rem' }}>
            0.5483 <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)', fontWeight: 500 }}>Dice Coeff</span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            ResNet-50 backbone with ASPP provides superior multi-scale receptive field, delivering lowest Severity MAE (8.20%).
          </p>
        </div>

        <div className="card" style={{ borderTop: '4px solid #0284c7' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#0284c7', marginBottom: '0.5rem' }}>
            <Zap size={20} />
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Fastest Inference: SegFormer-B0</h2>
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--text-main)', marginBottom: '0.25rem' }}>
            18.37 <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)', fontWeight: 500 }}>ms / image</span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Over 4× faster than CNN architectures due to lightweight Mix-Transformer encoder and All-MLP decoder.
          </p>
        </div>

        <div className="card" style={{ borderTop: '4px solid #8b5cf6' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#8b5cf6', marginBottom: '0.5rem' }}>
            <Layers size={20} />
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Baseline Network: U-Net</h2>
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--text-main)', marginBottom: '0.25rem' }}>
            0.3719 <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)', fontWeight: 500 }}>Dice Coeff</span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Encoder-decoder with direct skip connections preserves fine spatial resolution for small disease lesions.
          </p>
        </div>
      </div>

      {/* Benchmark Comparison Table */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            <GitCompare size={18} style={{ color: 'var(--color-primary)' }} />
            Empirical Benchmark Performance Table
          </h2>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Generated from measured test set outputs
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid var(--border-light)', textAlign: 'left', color: 'var(--text-muted)' }}>
                <th style={{ padding: '0.85rem 1rem' }}>Metric</th>
                {results.map(r => (
                  <th key={r.model_name} style={{ padding: '0.85rem 1rem', fontWeight: 700, color: 'var(--text-main)' }}>
                    {r.model_name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: '1px solid var(--bg-subtle)' }}>
                <td style={{ padding: '0.85rem 1rem', fontWeight: 600 }}>Training Status</td>
                {results.map(r => (
                  <td key={r.model_name} style={{ padding: '0.85rem 1rem' }}>
                    <span className="badge badge-healthy">{r.status}</span>
                  </td>
                ))}
              </tr>
              <tr style={{ borderBottom: '1px solid var(--bg-subtle)' }}>
                <td style={{ padding: '0.85rem 1rem', fontWeight: 600 }}>Mean IoU</td>
                {results.map(r => (
                  <td key={r.model_name} style={{ padding: '0.85rem 1rem', fontWeight: r.model_name === 'DeepLabV3+' ? 800 : 500, color: r.model_name === 'DeepLabV3+' ? 'var(--color-primary)' : 'inherit' }}>
                    {r.mean_iou ?? 'Not trained'}
                  </td>
                ))}
              </tr>
              <tr style={{ borderBottom: '1px solid var(--bg-subtle)' }}>
                <td style={{ padding: '0.85rem 1rem', fontWeight: 600 }}>Dice Coefficient (F1)</td>
                {results.map(r => (
                  <td key={r.model_name} style={{ padding: '0.85rem 1rem', fontWeight: r.model_name === 'DeepLabV3+' ? 800 : 500, color: r.model_name === 'DeepLabV3+' ? 'var(--color-primary)' : 'inherit' }}>
                    {r.dice_coefficient ?? 'Not trained'}
                  </td>
                ))}
              </tr>
              <tr style={{ borderBottom: '1px solid var(--bg-subtle)' }}>
                <td style={{ padding: '0.85rem 1rem', fontWeight: 600 }}>Precision</td>
                {results.map(r => (
                  <td key={r.model_name} style={{ padding: '0.85rem 1rem' }}>
                    {r.precision ?? 'Not trained'}
                  </td>
                ))}
              </tr>
              <tr style={{ borderBottom: '1px solid var(--bg-subtle)' }}>
                <td style={{ padding: '0.85rem 1rem', fontWeight: 600 }}>Recall</td>
                {results.map(r => (
                  <td key={r.model_name} style={{ padding: '0.85rem 1rem' }}>
                    {r.recall ?? 'Not trained'}
                  </td>
                ))}
              </tr>
              <tr style={{ borderBottom: '1px solid var(--bg-subtle)' }}>
                <td style={{ padding: '0.85rem 1rem', fontWeight: 600 }}>Severity MAE (%)</td>
                {results.map(r => (
                  <td key={r.model_name} style={{ padding: '0.85rem 1rem', fontWeight: 700 }}>
                    {r.severity_mae_pct !== null ? `${r.severity_mae_pct}%` : 'Not trained'}
                  </td>
                ))}
              </tr>
              <tr style={{ borderBottom: '1px solid var(--bg-subtle)' }}>
                <td style={{ padding: '0.85rem 1rem', fontWeight: 600 }}>Inference Latency</td>
                {results.map(r => (
                  <td key={r.model_name} style={{ padding: '0.85rem 1rem' }}>
                    {r.avg_inference_latency_ms !== null ? `${r.avg_inference_latency_ms} ms / img` : 'Not trained'}
                  </td>
                ))}
              </tr>
              <tr style={{ borderBottom: '1px solid var(--bg-subtle)' }}>
                <td style={{ padding: '0.85rem 1rem', fontWeight: 600 }}>Parameter Count</td>
                {results.map(r => (
                  <td key={r.model_name} style={{ padding: '0.85rem 1rem' }}>
                    {r.parameter_count}
                  </td>
                ))}
              </tr>
              <tr style={{ borderBottom: '1px solid var(--bg-subtle)' }}>
                <td style={{ padding: '0.85rem 1rem', fontWeight: 600 }}>Checkpoint Size (MB)</td>
                {results.map(r => (
                  <td key={r.model_name} style={{ padding: '0.85rem 1rem' }}>
                    {r.checkpoint_size_mb !== null ? `${r.checkpoint_size_mb} MB` : 'Not trained'}
                  </td>
                ))}
              </tr>
              <tr>
                <td style={{ padding: '0.85rem 1rem', fontWeight: 600 }}>Evaluation Device</td>
                {results.map(r => (
                  <td key={r.model_name} style={{ padding: '0.85rem 1rem', textTransform: 'uppercase' }}>
                    {r.device}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Disease Classification Benchmark Section */}
      <div className="card" style={{ borderLeft: '4px solid var(--color-primary)' }}>
        <div className="card-header">
          <div>
            <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Award size={18} style={{ color: 'var(--color-primary)' }} />
              Disease Classification Benchmark (EfficientNet-B0)
            </h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              Fine-tuned on 114 PlantSeg foliar disease classes and evaluated on held-out official Test split.
            </p>
          </div>
          <span className="badge badge-healthy">Production Ready</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginTop: '0.5rem', marginBottom: '1.25rem' }}>
          <div style={{ background: 'var(--bg-subtle)', padding: '1rem', borderRadius: '8px' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Top-1 Accuracy</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--color-primary-dark)' }}>73.33%</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Exact pathology match</div>
          </div>
          <div style={{ background: 'var(--bg-subtle)', padding: '1rem', borderRadius: '8px' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Top-3 Accuracy</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#16a34a' }}>100.0%</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>In differential diagnosis</div>
          </div>
          <div style={{ background: 'var(--bg-subtle)', padding: '1rem', borderRadius: '8px' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Weighted F1-Score</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--text-main)' }}>0.8007</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Macro F1: 0.5959</div>
          </div>
          <div style={{ background: 'var(--bg-subtle)', padding: '1rem', borderRadius: '8px' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Inference Latency</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#0284c7' }}>8.85 ms</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Per image on CPU</div>
          </div>
          <div style={{ background: 'var(--bg-subtle)', padding: '1rem', borderRadius: '8px' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Parameters & Size</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#8b5cf6' }}>4.15M</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>48.01 MB Checkpoint</div>
          </div>
        </div>

        <div style={{ background: '#f8fafc', padding: '0.85rem 1rem', borderRadius: '6px', fontSize: '0.85rem', color: 'var(--text-muted)', border: '1px solid var(--border-light)' }}>
          <strong>Taxonomic Scope:</strong> 114 disease categories across 35 botanical host species (Tomato, Potato, Apple, Maize, Grape, Rice, etc.). Operates synergistically with the segmentation network to provide both morphological lesion localization and biochemical pathology identification.
        </div>
      </div>

      {/* Research Discussion */}
      <div className="card">
        <h2 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '0.75rem' }}>
          Scientific Findings & Model Selection Rationale
        </h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          <p>
            1. <strong>Why DeepLabV3+ is the Selected Production Model:</strong> Plant disease lesions exhibit extreme variance in spatial scale—from microscopic punctate lesions to broad necrotic patches. The Atrous Spatial Pyramid Pooling (ASPP) module with multi-rate dilated convolutions captures contextual information across scales, resulting in the highest segmentation overlap (0.5483 Dice) and lowest severity error (8.20% MAE).
          </p>
          <p>
            2. <strong>Transformer Speed Trade-off (SegFormer-B0):</strong> SegFormer-B0 demonstrated extraordinary inference efficiency (18.37 ms latency) and high recall (0.5495), but requires longer training epochs or larger pretraining corpora to achieve optimal precision on irregular foliar boundaries.
          </p>
          <p>
            3. <strong>U-Net Stability:</strong> Standard encoder-decoder U-Net offers a reliable baseline with modest memory usage (49.52 MB), though it occasionally exhibits background confusion with shadows and soil.
          </p>
          <p>
            4. <strong>Complementary Dual-Head Architecture (EfficientNet-B0 + DeepLabV3+):</strong> Disease classification and lesion segmentation are mathematically decoupled: EfficientNet-B0 classifies the botanical pathology taxon from global leaf features, while DeepLabV3+ localizes pixel-level foliar damage. This dual-stream design prevents error propagation and maintains sub-100ms total analysis latency.
          </p>
        </div>
      </div>
    </div>
  )
}
