import { getAuthToken } from './supabase'

const BASE_URL = '' // Proxied by Vite to http://127.0.0.1:8000

/**
 * Fetch wrapper that attaches the Authorization: Bearer <token> header
 * at request time from the active Supabase session.
 * Preserves multipart/form-data boundary handling and all existing fetch options.
 */
async function authFetch(url, options = {}) {
  const token = await getAuthToken()
  const headers = new Headers(options.headers || {})

  if (token) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  return fetch(url, {
    ...options,
    headers
  })
}

export async function fetchHealth() {
  const res = await fetch(`${BASE_URL}/api/health`)
  if (!res.ok) throw new Error('Health check failed')
  return res.json()
}

export async function fetchModels() {
  const res = await fetch(`${BASE_URL}/api/models`)
  if (!res.ok) throw new Error('Failed to fetch models')
  return res.json()
}

export async function fetchModelComparison() {
  const res = await fetch(`${BASE_URL}/api/models/comparison`)
  if (!res.ok) throw new Error('Failed to fetch model comparison')
  return res.json()
}

export async function predictImage(file, modelName = null) {
  const formData = new FormData()
  formData.append('file', file)
  if (modelName) {
    formData.append('model_name', modelName)
  }

  const res = await authFetch(`${BASE_URL}/api/predict`, {
    method: 'POST',
    body: formData
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    throw new Error(errorData.detail || 'Analysis request failed')
  }

  return res.json()
}

export async function classifyImage(file) {
  const formData = new FormData()
  formData.append('file', file)

  const res = await authFetch(`${BASE_URL}/api/classify`, {
    method: 'POST',
    body: formData
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    throw new Error(errorData.detail || 'Classification failed')
  }

  return res.json()
}

export async function fetchPredictionHistory(page = 1, limit = 20, modelName = null, category = null, diseaseName = null) {
  const params = new URLSearchParams({ page, limit })
  if (modelName) params.append('model_name', modelName)
  if (category) params.append('severity_category', category)
  if (diseaseName) params.append('disease_name', diseaseName)

  const res = await authFetch(`${BASE_URL}/api/history?${params.toString()}`)
  if (!res.ok) throw new Error('Failed to fetch prediction history')
  return res.json()
}

export async function fetchPredictionDetail(id) {
  const res = await authFetch(`${BASE_URL}/api/history/${id}`)
  if (!res.ok) throw new Error('Failed to fetch prediction detail')
  return res.json()
}

export async function deletePrediction(id) {
  const res = await authFetch(`${BASE_URL}/api/history/${id}`, {
    method: 'DELETE'
  })
  if (!res.ok) throw new Error('Failed to delete prediction')
  return res.json()
}

export async function fetchAnalyticsSummary() {
  const res = await authFetch(`${BASE_URL}/api/analytics/summary`)
  if (!res.ok) throw new Error('Failed to fetch analytics')
  return res.json()
}

export async function fetchDatasetSummary() {
  const res = await fetch(`${BASE_URL}/api/dataset/summary`)
  if (!res.ok) throw new Error('Failed to fetch dataset summary')
  return res.json()
}

export async function fetchDatasetSamples() {
  const res = await fetch(`${BASE_URL}/api/dataset/samples`)
  if (!res.ok) throw new Error('Failed to fetch dataset samples')
  return res.json()
}
