import { useNavigate } from "react-router-dom"
import Logo from "../components/Logo"
import SamGuide from "../components/SamGuide"
import SectionCard from "../components/SectionCard"
import { practiceGroups } from "../data/practiceData"
import { loadPracticePlan } from "../utils/practicePlanner"

export default function PracticeHub() {
  const navigate = useNavigate()
  const plan = loadPracticePlan()

  const recommendedIds = new Set(plan?.recommendedGroupIds || [])
  const recommendedTarget = plan?.focusAreas?.[0]

  const orderedGroups = [
    ...practiceGroups.filter((group) => recommendedIds.has(group.id)),
    ...practiceGroups.filter((group) => !recommendedIds.has(group.id)),
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#dff7f2] via-[#e6f0ff] to-[#f5eaff] px-6 py-6">
      <div className="mx-auto max-w-6xl">
        <Logo />

        <div className="mt-6">
          <h1 className="text-4xl font-bold text-slate-900">Targeted Practice</h1>
          <p className="mt-3 max-w-3xl text-lg text-slate-600">
            Choose a sound group and practice one lesson at a time using minimal pairs.
          </p>
        </div>

        <SamGuide
          className="mt-6"
          text="If a sound feels tricky, I will suggest the best next lesson for you."
        />

        {recommendedTarget && (
          <SectionCard
            title="Recommended for you"
            subtitle="Based on your saved progress."
            className="mt-8 border-2 border-[#5B6CFF]/20"
          >
            <div className="mt-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-2xl font-bold text-slate-900">{recommendedTarget.groupTitle}</p>
                <p className="mt-2 text-slate-600">{plan.summary}</p>
              </div>

              <button
                onClick={() => navigate(`/practice/${recommendedTarget.groupId}/${recommendedTarget.lessonId}`)}
                className="rounded-2xl bg-[#5B6CFF] px-5 py-3 text-base font-semibold text-white transition hover:bg-[#4a57f0]"
              >
                Start recommended lesson
              </button>
            </div>
          </SectionCard>
        )}

        <div className="mt-8 grid gap-5 md:grid-cols-2 xl:grid-cols-3">
          {orderedGroups.map((group) => {
            const isRecommended = recommendedIds.has(group.id)

            return (
              <button
                key={group.id}
                onClick={() => navigate(`/practice/${group.id}`)}
                className="min-h-[180px] rounded-3xl bg-white p-6 text-left shadow-lg transition hover:-translate-y-1 hover:shadow-xl"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-semibold uppercase tracking-wide text-[#14b8a6]">
                    {group.subtitle}
                  </div>
                  {isRecommended && (
                    <span className="rounded-full bg-[#dff7f2] px-3 py-1 text-xs font-semibold text-slate-700">
                      Recommended
                    </span>
                  )}
                </div>

                <h2 className="mt-3 text-2xl font-bold text-slate-900">{group.title}</h2>
                <p className="mt-3 text-slate-600">{group.focus}</p>

                <div className="mt-5 text-sm font-medium text-slate-500">
                  {group.lessons.length} lessons available
                </div>
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}