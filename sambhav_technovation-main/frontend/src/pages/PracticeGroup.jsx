import { useNavigate, useParams } from "react-router-dom"
import Logo from "../components/Logo"
import SamGuide from "../components/SamGuide"
import SectionCard from "../components/SectionCard"
import PrimaryButton from "../components/PrimaryButton"
import { getPracticeGroupById } from "../data/practiceData"
import { loadPracticePlan } from "../utils/practicePlanner"

export default function PracticeGroup() {
  const { groupId } = useParams()
  const navigate = useNavigate()
  const plan = loadPracticePlan()

  const group = getPracticeGroupById(groupId)

  if (!group) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-white">
        <p className="text-slate-700">Practice group not found.</p>
      </div>
    )
  }

  const recommendedLessonIds = new Set(
    (plan?.focusAreas || [])
      .filter((item) => item.groupId === group.id)
      .map((item) => item.lessonId)
  )

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#dff7f2] via-[#e6f0ff] to-[#f5eaff] px-6 py-6">
      <div className="mx-auto max-w-6xl">
        <Logo />

        <button
          onClick={() => navigate("/practice")}
          className="mt-6 text-sm text-slate-600 underline"
        >
          Back to practice hub
        </button>

        <div className="mt-6">
          <h1 className="text-4xl font-bold text-slate-900">{group.title}</h1>
          <p className="mt-3 max-w-3xl text-lg text-slate-600">{group.focus}</p>
        </div>

        <SamGuide
          className="mt-6"
          text="Choose a lesson inside this sound group and I will guide you through it."
        />

        <div className="mt-8 grid gap-5 md:grid-cols-2">
          {group.lessons.map((lesson) => {
            const recommended = recommendedLessonIds.has(lesson.id)

            return (
              <SectionCard
                key={lesson.id}
                title={lesson.title}
                subtitle={lesson.focus}
                className={recommended ? "border-2 border-[#5B6CFF]/20" : ""}
              >
                <div className="mt-4 flex flex-wrap gap-2">
                  {lesson.words.slice(0, 4).map((pair, idx) => (
                    <span
                      key={`${pair.left}-${pair.right}-${idx}`}
                      className="rounded-full bg-slate-100 px-3 py-2 text-sm font-medium text-slate-700"
                    >
                      {pair.left} / {pair.right}
                    </span>
                  ))}
                </div>

                <div className="mt-5 flex flex-col gap-3 sm:flex-row">
                  <PrimaryButton
                    onClick={() => navigate(`/practice/${group.id}/${lesson.id}`)}
                    className="w-full sm:w-auto"
                  >
                    Start lesson
                  </PrimaryButton>

                  {recommended && (
                    <button
                      onClick={() => navigate(`/practice/${group.id}/${lesson.id}`)}
                      className="w-full rounded-2xl border border-[#5B6CFF] bg-white px-5 py-3 text-base font-semibold text-[#5B6CFF] transition hover:bg-slate-50 sm:w-auto"
                    >
                      Recommended
                    </button>
                  )}
                </div>
              </SectionCard>
            )
          })}
        </div>
      </div>
    </div>
  )
}