import React, { useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import LandingPage from './pages/LandingPage'
import Dashboard from './pages/Dashboard'
import AnalyzeImage from './pages/AnalyzeImage'
import ModelComparison from './pages/ModelComparison'
import PredictionHistory from './pages/PredictionHistory'
import DatasetExplorer from './pages/DatasetExplorer'
import AboutProject from './pages/AboutProject'
import { ensureAnonymousSession } from './services/supabase'

export default function App() {
  useEffect(() => {
    // Ensures a valid session exists without duplicating users or interrupting existing sessions
    ensureAnonymousSession().catch(err => {
      console.warn('Initial session check failed:', err)
    })
  }, [])

  return (
    <BrowserRouter>
      <div className="app-container">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/analyze" element={<AnalyzeImage />} />
            <Route path="/models" element={<ModelComparison />} />
            <Route path="/history" element={<PredictionHistory />} />
            <Route path="/dataset" element={<DatasetExplorer />} />
            <Route path="/about" element={<AboutProject />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  )
}
