import { useLocation, useNavigate } from "react-router-dom"
import Logo from "../components/Logo"
import SamGuide from "../components/SamGuide"
import SectionCard from "../components/SectionCard"
import PrimaryButton from "../components/PrimaryButton"
import ProgressBar from "../components/ProgressBar"
import { loadAppState, getTopWeakPhonemes } from "../utils/storage"
import { buildPracticePlan } from "../utils/practicePlanner"

function loadStoredReport() {
  if (typeof window === "undefined") return {}
  try {
    return JSON.parse(window.localStorage.getItem("sambhav_last_report") || "{}")
  } catch {
    return {}
  }
}

export default function Report() {
  const location = useLocation()
  const navigate = useNavigate()

  const session = location.state || loadStoredReport()
  const appState = loadAppState()

  const {
    results = [],
    weakPhonemes = {},
    groupScores = {},
    diagnosis = false,
    practice = false,
    title = "Session Report",
  } = session || {}

  const total = results.length
  const correct = results.filter((r) => r.correct).length
  const accuracy = total ? Math.round((correct / total) * 100) : 0

  const plan = buildPracticePlan({ groupScores, weakPhonemes })
  const topWeak = getTopWeakPhonemes(weakPhonemes, 4)
  const recentSessions = appState.progress?.recentSessions || []

  const startRecommended = () => {
    const first = plan.focusAreas[0]
    if (!first) {
      navigate("/practice")
      return
    }
    navigate(`/practice/${first.groupId}/${first.lessonId}`)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#dff7f2] via-[#e6f0ff] to-[#f5eaff] px-6 py-6">
      <div className="mx-auto max-w-6xl">
        <Logo />

        <div className="mt-6">
          <p className="text-sm font-semibold uppercase tracking-wide text-[#5B6CFF]">
            {title}
          </p>
          <h1 className="mt-2 text-4xl font-bold text-slate-900">
            {diagnosis ? "Screening Report" : practice ? "Practice Report" : "Lesson Report"}
          </h1>
          <p className="mt-3 max-w-3xl text-lg text-slate-600">
            This report is saved locally on this device.
          </p>
        </div>

        <SamGuide
          className="mt-6"
          text={
            plan.focusAreas.length > 0
              ? "I can show you the next best lesson based on the sounds that need more practice."
              : "Great work. Your progress is saved locally and you can continue anytime."
          }
        />

        <div className="mt-8 grid gap-5 md:grid-cols-3">
          <SectionCard title="Accuracy" subtitle="Current session">
            <p className="mt-4 text-4xl font-bold text-slate-900">{accuracy}%</p>
          </SectionCard>

          <SectionCard title="Words / items" subtitle="Current session">
            <p className="mt-4 text-4xl font-bold text-slate-900">{total}</p>
          </SectionCard>

          <SectionCard title="Correct" subtitle="Current session">
            <p className="mt-4 text-4xl font-bold text-slate-900">{correct}</p>
          </SectionCard>
        </div>

        <div className="mt-5">
          <ProgressBar
            label="Saved local progress"
            value={Math.min(
              100,
              (appState.progress.lessonsCompleted || 0) * 18 +
                (appState.progress.practiceSessions || 0) * 14 +
                (appState.progress.diagnosisRuns || 0) * 12 +
                Math.min(20, Math.floor((appState.progress.totalWordsPracticed || 0) / 3))
            )}
            helper="Stored only on this device."
          />
        </div>

        {topWeak.length > 0 && (
          <SectionCard
            title="Focus sounds"
            subtitle="Most common sounds that need more practice."
            className="mt-5"
          >
            <div className="mt-4 flex flex-wrap gap-2">
              {topWeak.map(([phoneme, count]) => (
                <span
                  key={phoneme}
                  className="rounded-full bg-[#dff7f2] px-3 py-2 text-sm font-semibold text-slate-800"
                >
                  /{phoneme}/ × {count}
                </span>
              ))}
            </div>
          </SectionCard>
        )}

        {results.length > 0 && (
          <SectionCard title="Session details" subtitle="What happened in this run." className="mt-5">
            <div className="mt-4 space-y-3">
              {results.map((item, i) => (
                <div key={i} className="rounded-2xl bg-slate-50 px-4 py-3 text-slate-700">
                  <div className="font-semibold text-slate-900">
                    {item.word || item.target}
                  </div>
                  <div className="mt-1">
                    {item.correct ? "Correct" : "Needs more practice"}
                    {typeof item.attempts === "number" ? ` · ${item.attempts} attempt(s)` : ""}
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>
        )}

        {plan.focusAreas.length > 0 && (
          <SectionCard title="Recommended next practice" subtitle="Based on this report." className="mt-5">
            <p className="mt-4 text-slate-700">{plan.summary}</p>

            <div className="mt-5 grid gap-4 md:grid-cols-2">
              {plan.focusAreas.map((area) => (
                <div key={`${area.groupId}-${area.lessonId}`} className="rounded-3xl bg-slate-50 p-5">
                  <h3 className="text-2xl font-bold text-slate-900">{area.groupTitle}</h3>
                  <p className="mt-2 text-slate-600">{area.reasons.join(" ")}</p>
                  <button
                    onClick={() => navigate(`/practice/${area.groupId}/${area.lessonId}`)}
                    className="mt-4 rounded-2xl bg-[#5B6CFF] px-5 py-3 font-semibold text-white transition hover:bg-[#4a57f0]"
                  >
                    Start this lesson
                  </button>
                </div>
              ))}
            </div>
          </SectionCard>
        )}

        {recentSessions.length > 0 && (
          <SectionCard title="Recent local sessions" subtitle="Stored only on this device." className="mt-5">
            <div className="mt-4 space-y-3">
              {recentSessions.slice(0, 4).map((sessionItem) => (
                <div key={sessionItem.id} className="rounded-2xl bg-slate-50 px-4 py-3 text-slate-700">
                  <div className="font-semibold text-slate-900">{sessionItem.title}</div>
                  <div className="mt-1 text-sm text-slate-600">
                    {sessionItem.accuracy}% accuracy · {sessionItem.wordsPracticed} items
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>
        )}

        <div className="mt-6 flex flex-wrap gap-3">
          <PrimaryButton onClick={() => navigate("/dashboard")}>
            Back to Dashboard
          </PrimaryButton>

          <button
            onClick={() => navigate("/practice")}
            className="rounded-2xl border border-slate-200 bg-white px-5 py-3 text-base font-semibold text-slate-700 transition hover:bg-slate-50"
          >
            Practice Hub
          </button>

          {plan.focusAreas.length > 0 && (
            <button
              onClick={startRecommended}
              className="rounded-2xl border border-[#5B6CFF] bg-white px-5 py-3 text-base font-semibold text-[#5B6CFF] transition hover:bg-slate-50"
            >
              Continue recommended practice
            </button>
          )}
        </div>
      </div>
    </div>
  )
}