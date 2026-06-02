import { useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import Logo from "../components/Logo"
import SamGuide from "../components/SamGuide"
import ProgressBar from "../components/ProgressBar"
import SectionCard from "../components/SectionCard"
import PrimaryButton from "../components/PrimaryButton"
import OnboardingModal from "../components/OnboardingModal"
import {
  loadAppState,
  markFirstVisitComplete,
  getTopWeakPhonemes,
} from "../utils/storage"

export default function Dashboard() {
  const navigate = useNavigate()
  const [appState, setAppState] = useState(() => loadAppState())
  const [showIntro, setShowIntro] = useState(() => loadAppState().firstRun)

  const progress = appState.progress || {}

  const hasAnyActivity =
    (progress.lessonsCompleted || 0) > 0 ||
    (progress.practiceSessions || 0) > 0 ||
    (progress.diagnosisRuns || 0) > 0 ||
    (progress.totalWordsPracticed || 0) > 0

  const topWeak = useMemo(
    () => getTopWeakPhonemes(progress.weakPhonemes || {}, 4),
    [progress.weakPhonemes]
  )

  const overallJourney = Math.min(
    100,
    (progress.lessonsCompleted || 0) * 18 +
      (progress.practiceSessions || 0) * 14 +
      (progress.diagnosisRuns || 0) * 12 +
      Math.min(20, Math.floor((progress.totalWordsPracticed || 0) / 3))
  )

  const beginRoute = (path) => {
    const next = markFirstVisitComplete()
    setAppState(next)
    setShowIntro(false)
    navigate(path)
  }

  const dismissIntro = () => {
    const next = markFirstVisitComplete()
    setAppState(next)
    setShowIntro(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#dff7f2] via-[#e6f0ff] to-[#f5eaff] px-6 py-6">
      <OnboardingModal
        open={showIntro}
        onStartLesson={() => beginRoute("/lesson")}
        onPractice={() => beginRoute("/practice")}
        onDiagnosis={() => beginRoute("/diagnosis")}
        onDismiss={dismissIntro}
      />

      <div className="mx-auto max-w-6xl">
        <Logo />

        <div className="mt-6">
          <h1 className="text-4xl font-bold text-slate-900">
            {hasAnyActivity ? "Welcome back" : "Welcome to Sambhav"}
          </h1>
          <p className="mt-2 max-w-2xl text-lg text-slate-600">
            {hasAnyActivity
              ? "Your progress is saved locally on this device."
              : "Start with one guided lesson. Sam will lead you step by step."}
          </p>
        </div>

        <SamGuide
          className="mt-6"
          text={
            hasAnyActivity
              ? "Pick up where you left off, or choose targeted practice if a sound feels hard."
              : "Choose Lesson to begin, or open Practice Mode if you want minimal pairs first."
          }
        />

        <div className="mt-8 grid gap-5 lg:grid-cols-3">
          <SectionCard
            title="Voice Coach Session"
            subtitle="Continue your guided practice."
            className="lg:col-span-2 min-h-[220px] bg-gradient-to-r from-[#6d5efc] to-[#5b6cff] text-white"
          >
            <p className="mt-4 max-w-xl text-white/90">
              Work through one word at a time with Sam. Feedback stays clear and simple, and the
              next step is always easy to find.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <PrimaryButton onClick={() => beginRoute("/lesson")} className="bg-white text-slate-900 hover:bg-slate-100">
                Start Lesson
              </PrimaryButton>
              <PrimaryButton onClick={() => beginRoute("/practice")} className="bg-[#14b8a6] hover:bg-[#0f9e8f]">
                Targeted Practice
              </PrimaryButton>
            </div>
          </SectionCard>

          <SectionCard
            title="Current Summary"
            subtitle="Stored locally."
            className="min-h-[220px]"
          >
            <div className="mt-5 space-y-4">
              <div>
                <p className="text-sm text-slate-500">Accuracy</p>
                <p className="text-3xl font-bold text-slate-900">
                  {progress.averageAccuracy || 0}%
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-500">Streak</p>
                <p className="text-3xl font-bold text-slate-900">
                  {progress.streakDays || 0} days
                </p>
              </div>
            </div>
          </SectionCard>
        </div>

        <div className="mt-5">
          <ProgressBar
            label="Journey progress"
            value={overallJourney}
            helper={
              hasAnyActivity
                ? "Based on completed lessons, practice, and screening."
                : "Your journey starts here."
            }
          />
        </div>

        <div className="mt-5 grid gap-5 md:grid-cols-2 xl:grid-cols-4">
          <SectionCard title="Lessons" subtitle="Completed sessions">
            <p className="mt-4 text-4xl font-bold text-slate-900">
              {progress.lessonsCompleted || 0}
            </p>
          </SectionCard>

          <SectionCard title="Practice" subtitle="Targeted sessions">
            <p className="mt-4 text-4xl font-bold text-slate-900">
              {progress.practiceSessions || 0}
            </p>
          </SectionCard>

          <SectionCard title="Screening" subtitle="Runs completed">
            <p className="mt-4 text-4xl font-bold text-slate-900">
              {progress.diagnosisRuns || 0}
            </p>
          </SectionCard>

          <SectionCard title="Words" subtitle="Practiced locally">
            <p className="mt-4 text-4xl font-bold text-slate-900">
              {progress.totalWordsPracticed || 0}
            </p>
          </SectionCard>
        </div>

        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <SectionCard title="Sam's guidance" subtitle="Start with a clear path">
            <div className="mt-4 space-y-3 text-slate-700">
              <p>1. Start with a lesson if you are new.</p>
              <p>2. Use targeted practice for tricky sounds.</p>
              <p>3. Run screening to see which groups need more support.</p>
            </div>
          </SectionCard>

          <SectionCard title="Focus sounds" subtitle="Most common weak phonemes">
            <div className="mt-4 flex flex-wrap gap-2">
              {topWeak.length > 0 ? (
                topWeak.map(([phoneme, count]) => (
                  <span
                    key={phoneme}
                    className="rounded-full bg-[#dff7f2] px-3 py-2 text-sm font-semibold text-slate-800"
                  >
                    /{phoneme}/ × {count}
                  </span>
                ))
              ) : (
                <p className="text-slate-600">
                  No saved weak sounds yet. Start a lesson to build progress.
                </p>
              )}
            </div>
          </SectionCard>
        </div>

        <div className="mt-6 grid gap-5 md:grid-cols-2 xl:grid-cols-4">
          <button
            onClick={() => beginRoute("/lesson")}
            className="min-h-[140px] rounded-3xl bg-white p-6 text-left shadow-lg transition hover:-translate-y-1 hover:shadow-xl"
          >
            <div className="text-sm font-semibold uppercase tracking-wide text-[#5B6CFF]">
              Core learning
            </div>
            <div className="mt-2 text-2xl font-bold text-slate-900">Start Lesson</div>
            <p className="mt-2 text-slate-600">
              One word at a time with clear feedback.
            </p>
          </button>

          <button
            onClick={() => beginRoute("/practice")}
            className="min-h-[140px] rounded-3xl bg-white p-6 text-left shadow-lg transition hover:-translate-y-1 hover:shadow-xl"
          >
            <div className="text-sm font-semibold uppercase tracking-wide text-[#14b8a6]">
              Targeted practice
            </div>
            <div className="mt-2 text-2xl font-bold text-slate-900">Practice Mode</div>
            <p className="mt-2 text-slate-600">
              Minimal pairs and phoneme-specific lessons.
            </p>
          </button>

          <button
            onClick={() => beginRoute("/diagnosis")}
            className="min-h-[140px] rounded-3xl bg-white p-6 text-left shadow-lg transition hover:-translate-y-1 hover:shadow-xl"
          >
            <div className="text-sm font-semibold uppercase tracking-wide text-[#7c3aed]">
              Screening
            </div>
            <div className="mt-2 text-2xl font-bold text-slate-900">Diagnosis Mode</div>
            <p className="mt-2 text-slate-600">
              See which sound groups need attention.
            </p>
          </button>

          <button
            onClick={() => beginRoute("/about")}
            className="min-h-[140px] rounded-3xl bg-white p-6 text-left shadow-lg transition hover:-translate-y-1 hover:shadow-xl"
          >
            <div className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              Learn more
            </div>
            <div className="mt-2 text-2xl font-bold text-slate-900">About Sambhav</div>
            <p className="mt-2 text-slate-600">
              Read about the mission and how it works.
            </p>
          </button>
        </div>
      </div>
    </div>
  )
}