const STORAGE_KEY = "sambhav_app_state_v2"
const SESSIONS_KEY = "sambhav_sessions_v2"

export function createDefaultAppState() {
  return {
    firstRun: true,
    profile: {
      name: "",
    },
    progress: {
      lessonsCompleted: 0,
      practiceSessions: 0,
      diagnosisRuns: 0,
      totalWordsPracticed: 0,
      averageAccuracy: 0,
      weakPhonemes: {},
      recentSessions: [],
      lastRoute: "/dashboard",
      lastUpdatedAt: null,
    },
    ui: {
      theme: "soft",
      hasSeenIntro: false,
    },
  }
}

export function loadAppState() {
  if (typeof window === "undefined") return createDefaultAppState()

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return createDefaultAppState()

    const parsed = JSON.parse(raw)
    return {
      ...createDefaultAppState(),
      ...parsed,
      profile: {
        ...createDefaultAppState().profile,
        ...(parsed.profile || {}),
      },
      progress: {
        ...createDefaultAppState().progress,
        ...(parsed.progress || {}),
        weakPhonemes: {
          ...createDefaultAppState().progress.weakPhonemes,
          ...((parsed.progress && parsed.progress.weakPhonemes) || {}),
        },
      },
      ui: {
        ...createDefaultAppState().ui,
        ...(parsed.ui || {}),
      },
    }
  } catch {
    return createDefaultAppState()
  }
}

export function saveAppState(state) {
  if (typeof window === "undefined") return

  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
}

export function updateAppState(updater) {
  const current = loadAppState()
  const next = typeof updater === "function" ? updater(current) : { ...current, ...updater }
  saveAppState(next)
  return next
}

export function markFirstVisitComplete() {
  return updateAppState((state) => ({
    ...state,
    firstRun: false,
    progress: {
      ...state.progress,
      lastUpdatedAt: new Date().toISOString(),
    },
    ui: {
      ...state.ui,
      hasSeenIntro: true,
    },
  }))
}

export function recordSession(session) {
  const current = loadAppState()
  const sessions = loadSessions()

  const nextSession = {
    id: session.id || `${Date.now()}`,
    type: session.type || "lesson",
    title: session.title || "Session",
    accuracy: Number(session.accuracy || 0),
    wordsPracticed: Number(session.wordsPracticed || 0),
    weakPhonemes: session.weakPhonemes || {},
    createdAt: session.createdAt || new Date().toISOString(),
  }

  const nextSessions = [nextSession, ...sessions].slice(0, 20)

  const updatedWeakPhonemes = { ...current.progress.weakPhonemes }
  Object.entries(nextSession.weakPhonemes || {}).forEach(([phoneme, count]) => {
    updatedWeakPhonemes[phoneme] = (updatedWeakPhonemes[phoneme] || 0) + Number(count || 0)
  })

  const totalSessions = nextSessions.length
  const averageAccuracy =
    totalSessions > 0
      ? Math.round(
          nextSessions.reduce((sum, item) => sum + Number(item.accuracy || 0), 0) / totalSessions
        )
      : 0

  const nextState = {
    ...current,
    firstRun: false,
    progress: {
      ...current.progress,
      lessonsCompleted:
        current.progress.lessonsCompleted + (nextSession.type === "lesson" ? 1 : 0),
      practiceSessions:
        current.progress.practiceSessions + (nextSession.type === "practice" ? 1 : 0),
      diagnosisRuns:
        current.progress.diagnosisRuns + (nextSession.type === "diagnosis" ? 1 : 0),
      totalWordsPracticed: current.progress.totalWordsPracticed + nextSession.wordsPracticed,
      averageAccuracy,
      weakPhonemes: updatedWeakPhonemes,
      recentSessions: nextSessions,
      lastUpdatedAt: nextSession.createdAt,
    },
  }

  saveAppState(nextState)
  saveSessions(nextSessions)

  return nextState
}

export function loadSessions() {
  if (typeof window === "undefined") return []

  try {
    const raw = window.localStorage.getItem(SESSIONS_KEY)
    if (!raw) return []

    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function saveSessions(sessions) {
  if (typeof window === "undefined") return
  window.localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions))
}

export function resetAppState() {
  if (typeof window === "undefined") return

  window.localStorage.removeItem(STORAGE_KEY)
  window.localStorage.removeItem(SESSIONS_KEY)
}

export function getTopWeakPhonemes(weakPhonemes, limit = 3) {
  return Object.entries(weakPhonemes || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
}