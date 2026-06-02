import logo from "../assets/logo.png"

export default function Logo({ size = 48, showName = true, showBackHome = true }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <img src={logo} alt="Sambhav" width={size} height={size} className="block" />
        {showName && <span className="text-xl font-bold text-slate-900">Sambhav</span>}
      </div>

      {showBackHome && (
        <a
          href="/"
          className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-[#7c3aed] hover:text-[#7c3aed]"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
            <path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Back to Home
        </a>
      )}
    </div>
  )
}