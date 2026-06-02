import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import Logo from "../components/Logo"
import SamGuide from "../components/SamGuide"

export default function Splash() {
  const navigate = useNavigate()
  const [slideIndex, setSlideIndex] = useState(0)

  const slides = [
    {
      title: "Every child deserves to be heard",
      subtitle: "Speech and learning support that feels calm, clear, and personal.",
    },
    {
      title: "AI-powered phoneme coaching",
      subtitle: "Sam helps children practice sounds one step at a time.",
    },
    {
      title: "Built for accessibility",
      subtitle: "Designed for children with dyslexia and phonological challenges.",
    },
  ]

  useEffect(() => {
    const interval = setInterval(() => {
      setSlideIndex((prev) => (prev + 1) % slides.length)
    }, 3500)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#dff7f2] via-[#e6f0ff] to-[#f5eaff] px-6 py-6">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col justify-between">
        <div className="pt-2">
          <Logo />
        </div>

        <div className="flex flex-1 flex-col items-center justify-center text-center">
          <div className="max-w-3xl rounded-3xl bg-white/80 px-8 py-10 shadow-xl backdrop-blur">
            <p className="text-sm font-semibold uppercase tracking-wide text-[#5B6CFF]">
              Speech. Learning. Possibility.
            </p>

            <h1 className="mt-4 text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">
              {slides[slideIndex].title}
            </h1>

            <p className="mx-auto mt-4 max-w-2xl text-lg text-slate-600">
              {slides[slideIndex].subtitle}
            </p>

            <div className="mt-6 flex justify-center gap-2">
              {slides.map((_, i) => (
                <span
                  key={i}
                  className={`h-2.5 w-2.5 rounded-full transition ${
                    i === slideIndex ? "bg-[#5B6CFF]" : "bg-slate-300"
                  }`}
                />
              ))}
            </div>
          </div>

          <SamGuide
            className="mt-8"
            text="I will guide you through lessons, practice, and diagnosis."
          />
        </div>

        <div className="pb-2">
          <div className="mx-auto flex max-w-md flex-col gap-3">
            <button
              onClick={() => navigate("/login")}
              className="rounded-2xl bg-[#5B6CFF] px-6 py-4 text-lg font-semibold text-white shadow-lg transition hover:bg-[#4a57f0]"
            >
              Get Started
            </button>

            <button
              onClick={() => navigate("/about")}
              className="rounded-2xl border border-slate-300 bg-white px-6 py-4 text-base font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
            >
              Learn more about Sambhav
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}