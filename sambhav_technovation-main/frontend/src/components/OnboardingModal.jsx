export default function OnboardingModal({
    open,
    onStartLesson,
    onPractice,
    onDiagnosis,
    onDismiss,
  }) {
    if (!open) return null
  
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 px-6">
        <div className="w-full max-w-2xl rounded-3xl bg-white p-8 shadow-2xl">
          <p className="text-sm font-semibold uppercase tracking-wide text-[#5B6CFF]">
            Start here
          </p>
  
          <h2 className="mt-2 text-3xl font-bold text-slate-900">
            Welcome to Sambhav
          </h2>
  
          <p className="mt-3 text-lg text-slate-600">
            I will guide you through lessons, practice, and screening. Everything is saved locally on this device.
          </p>
  
          <div className="mt-6 grid gap-4 md:grid-cols-3">
            <div className="rounded-2xl bg-slate-50 p-4">
              <p className="text-sm font-semibold text-slate-900">1. Lesson</p>
              <p className="mt-2 text-sm text-slate-600">
                Start with one word at a time and get clear feedback.
              </p>
            </div>
  
            <div className="rounded-2xl bg-slate-50 p-4">
              <p className="text-sm font-semibold text-slate-900">2. Practice</p>
              <p className="mt-2 text-sm text-slate-600">
                Work on minimal pairs for sounds that need more attention.
              </p>
            </div>
  
            <div className="rounded-2xl bg-slate-50 p-4">
              <p className="text-sm font-semibold text-slate-900">3. Screening</p>
              <p className="mt-2 text-sm text-slate-600">
                Check which sound groups may need extra support.
              </p>
            </div>
          </div>
  
          <div className="mt-8 flex flex-col gap-3 sm:flex-row">
            <button
              onClick={onStartLesson}
              className="rounded-2xl bg-[#5B6CFF] px-5 py-3 text-base font-semibold text-white transition hover:bg-[#4a57f0]"
            >
              Start Lesson
            </button>
  
            <button
              onClick={onPractice}
              className="rounded-2xl border border-slate-200 bg-white px-5 py-3 text-base font-semibold text-slate-700 transition hover:bg-slate-50"
            >
              Practice Mode
            </button>
  
            <button
              onClick={onDiagnosis}
              className="rounded-2xl border border-slate-200 bg-white px-5 py-3 text-base font-semibold text-slate-700 transition hover:bg-slate-50"
            >
              Screening
            </button>
  
            <button
              onClick={onDismiss}
              className="rounded-2xl px-5 py-3 text-base font-semibold text-slate-500 transition hover:text-slate-700"
            >
              Maybe later
            </button>
          </div>
        </div>
      </div>
    )
  }