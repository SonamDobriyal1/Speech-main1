import { practiceGroups } from "../data/practiceData"

const PHONEME_TO_GROUP = {
  b: "bp",
  p: "bp",
  r: "rl",
  l: "rl",
  th: "th",
  dh: "th",
  k: "kg",
  g: "kg",
}

export function buildPracticePlan({ groupScores = {}, weakPhonemes = {} } = {}) {
  const scoreMap = new Map()
  const reasonsMap = new Map()

  const addReason = (groupId, reason) => {
    const current = reasonsMap.get(groupId) || []
    if (!current.includes(reason)) {
      reasonsMap.set(groupId, [...current, reason])
    }
  }

  for (const [groupId, data] of Object.entries(groupScores || {})) {
    const total = Number(data?.total || 0)
    const correct = Number(data?.correct || 0)
    const accuracy = total > 0 ? correct / total : 1

    if (accuracy < 0.85) {
      scoreMap.set(groupId, Math.max(scoreMap.get(groupId) || 0, Math.round((1 - accuracy) * 100)))
      const group = practiceGroups.find((g) => g.id === groupId)
      addReason(groupId, `Lower accuracy in ${group?.title || groupId}.`)
    }
  }

  for (const [phoneme, count] of Object.entries(weakPhonemes || {})) {
    const groupId = PHONEME_TO_GROUP[phoneme]
    if (!groupId) continue

    scoreMap.set(groupId, Math.max(scoreMap.get(groupId) || 0, 40 + count * 10))
    addReason(groupId, `Repeated trouble with /${phoneme}/.`)
  }

  const recommendedGroupIds = [...scoreMap.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([groupId]) => groupId)

  const focusAreas = recommendedGroupIds
    .map((groupId) => {
      const group = practiceGroups.find((g) => g.id === groupId)
      const lesson = group?.lessons?.[0] || null

      if (!group || !lesson) return null

      return {
        groupId,
        groupTitle: group.title,
        lessonId: lesson.id,
        lessonTitle: lesson.title,
        reasons: reasonsMap.get(groupId) || [],
      }
    })
    .filter(Boolean)

  const summary =
    focusAreas.length > 0
      ? `Recommended focus: ${focusAreas.map((area) => area.groupTitle).join(", ")}.`
      : "No additional practice recommended right now."

  return {
    recommendedGroupIds,
    focusAreas,
    summary,
    generatedAt: new Date().toISOString(),
  }
}

export function savePracticePlan(plan) {
  if (typeof window === "undefined") return
  localStorage.setItem("sambhav_practice_plan", JSON.stringify(plan))
}

export function loadPracticePlan() {
  if (typeof window === "undefined") {
    return {
      recommendedGroupIds: [],
      focusAreas: [],
      summary: "",
      generatedAt: null,
    }
  }

  try {
    const raw = localStorage.getItem("sambhav_practice_plan")
    if (!raw) {
      return {
        recommendedGroupIds: [],
        focusAreas: [],
        summary: "",
        generatedAt: null,
      }
    }
    return JSON.parse(raw)
  } catch {
    return {
      recommendedGroupIds: [],
      focusAreas: [],
      summary: "",
      generatedAt: null,
    }
  }
}