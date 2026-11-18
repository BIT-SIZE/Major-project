import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || ''
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || ''

/**
 * Checks whether valid Supabase frontend credentials are configured.
 * Validates that placeholder values have been replaced with real credentials.
 */
export const isSupabaseConfigured = Boolean(
  supabaseUrl &&
  supabaseAnonKey &&
  supabaseAnonKey !== 'your-supabase-publishable-anon-key-here' &&
  !supabaseAnonKey.startsWith('your-')
)

/**
 * Single canonical Supabase client for browser authentication and public queries.
 * Initialized only when valid configuration is present to avoid runtime crashes.
 */
export const supabase = isSupabaseConfigured
  ? createClient(supabaseUrl, supabaseAnonKey, {
      auth: {
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: true
      }
    })
  : null

// In-flight singleton promise to prevent concurrent calls from spawning multiple anonymous users
let authInitPromise = null
let lastAuthFailureTime = 0
const AUTH_RETRY_COOLDOWN_MS = 30000 // 30-second cooldown after failure to avoid infinite retry loops

/**
 * Ensures an active session exists.
 * - Reuses existing valid sessions (anonymous or registered).
 * - Deduplicates concurrent calls (e.g., React 19 StrictMode double mount).
 * - Avoids hammering GoTrue if anonymous auth fails (30s cooldown).
 * - Preserves existing registered users without accidental sign-out.
 */
export async function ensureAnonymousSession() {
  if (!isSupabaseConfigured || !supabase) {
    return null
  }

  // Deduplicate in-flight initialization
  if (authInitPromise) {
    return authInitPromise
  }

  // Enforce cooldown if a recent attempt failed
  if (Date.now() - lastAuthFailureTime < AUTH_RETRY_COOLDOWN_MS) {
    return null
  }

  authInitPromise = (async () => {
    try {
      // 1. Check existing session first (re-uses existing anonymous or registered user)
      const { data: { session }, error: sessionError } = await supabase.auth.getSession()
      if (sessionError) {
        console.warn('Supabase getSession warning:', sessionError.message)
      }

      if (session?.access_token) {
        return session
      }

      // 2. No session found; initiate anonymous sign-in
      const { data, error } = await supabase.auth.signInAnonymously()
      if (error) {
        lastAuthFailureTime = Date.now()
        console.warn('Supabase anonymous sign-in failed:', error.message)
        return null
      }

      return data?.session || null
    } catch (err) {
      lastAuthFailureTime = Date.now()
      console.warn('Failed to ensure anonymous session:', err)
      return null
    } finally {
      authInitPromise = null
    }
  })()

  return authInitPromise
}

/**
 * Retrieves the current JWT access token for authenticating FastAPI requests.
 * Refreshes near-expired tokens automatically and avoids storing static tokens.
 */
export async function getAuthToken() {
  if (!isSupabaseConfigured || !supabase) {
    return null
  }

  try {
    let { data: { session } } = await supabase.auth.getSession()

    // If no session exists yet, attempt to ensure an anonymous session
    if (!session?.access_token) {
      session = await ensureAnonymousSession()
    }

    if (!session?.access_token) {
      return null
    }

    // Refresh if expiring within the next 60 seconds
    if (session.expires_at) {
      const nowSeconds = Math.floor(Date.now() / 1000)
      if (session.expires_at - nowSeconds < 60) {
        const { data: refreshed, error: refreshErr } = await supabase.auth.refreshSession()
        if (!refreshErr && refreshed?.session) {
          session = refreshed.session
        }
      }
    }

    return session.access_token
  } catch (err) {
    console.warn('Failed to retrieve authentication token:', err)
    return null
  }
}

/**
 * Links email and password credentials to an anonymous user,
 * upgrading them to a permanent account while retaining the exact same user ID and records.
 */
export async function upgradeAnonymousAccount(email, password) {
  if (!isSupabaseConfigured || !supabase) {
    throw new Error('Supabase client is not configured.')
  }
  const { data, error } = await supabase.auth.updateUser({ email, password })
  if (error) {
    throw error
  }
  return data
}
