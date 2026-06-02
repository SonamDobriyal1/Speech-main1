import { useNavigate } from "react-router-dom"
import Logo from "../components/Logo"
import SamGuide from "../components/SamGuide"
import SectionCard from "../components/SectionCard"

export default function About() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#dff7f2] via-[#e6f0ff] to-[#f5eaff] px-6 py-6">
      <div className="mx-auto max-w-5xl">
        <Logo />

        <button
          onClick={() => navigate("/")}
          className="mt-6 text-sm text-slate-600 underline"
        >
          Back
        </button>

        <div className="mt-6">
          <h1 className="text-4xl font-bold text-slate-900">About Sambhav</h1>
          <p className="mt-3 max-w-3xl text-lg text-slate-600">
            Sambhav is an AI-powered voice learning platform designed to support children with
            dyslexia, dysgraphia, and phonological challenges. It focuses on sound-level learning
            and guided practice.
          </p>
        </div>

        <SamGuide
          className="mt-6"
          text="I help children practice one sound at a time and keep their progress saved locally."
        />

        <div className="mt-8 grid gap-5 md:grid-cols-2">
          <SectionCard
            title="The problem"
            subtitle="Many children need more targeted support than standard reading tools provide."
          >
            <p className="mt-4 text-slate-700">
              Traditional intervention is often expensive, hard to access, and not personalized.
              Children can be left without the right kind of practice for the exact sounds they
              struggle with.
            </p>
          </SectionCard>

          <SectionCard
            title="Our approach"
            subtitle="Sambhav uses speech recognition and phoneme-level feedback."
          >
            <p className="mt-4 text-slate-700">
              Instead of only marking answers as right or wrong, Sambhav identifies which sound
              needs attention and guides the learner through targeted practice.
            </p>
          </SectionCard>

          <SectionCard
            title="Why it matters"
            subtitle="Early support can improve confidence and reading outcomes."
          >
            <p className="mt-4 text-slate-700">
              The app is designed to be clear, accessible, and easy to follow for young learners.
              Progress is stored locally on the device for privacy.
            </p>
          </SectionCard>

          <SectionCard
            title="What Sambhav includes"
            subtitle="A guided learning experience for children and families."
          >
            <ul className="mt-4 space-y-2 text-slate-700">
              <li>Lesson mode with voice feedback</li>
              <li>Targeted practice using minimal pairs</li>
              <li>Diagnosis mode for screening patterns</li>
              <li>Local progress storage</li>
            </ul>
          </SectionCard>
        </div>

        <div className="mt-8 flex justify-center">
          <button
            onClick={() => navigate("/dashboard")}
            className="rounded-2xl bg-[#5B6CFF] px-6 py-4 text-lg font-semibold text-white shadow-lg transition hover:bg-[#4a57f0]"
          >
            Start using Sambhav
          </button>
        </div>
      </div>
    </div>
  )
}