export default function ProgressBar({ value = 0, label, helper }) {
    const safeValue = Math.max(0, Math.min(100, Number(value) || 0))
  
    return (
      <div className="rounded-2xl bg-white p-4 shadow">
        {label && <p className="text-sm font-medium text-slate-600">{label}</p>}
        <div className="mt-3 h-3 w-full rounded-full bg-slate-200">
          <div
            className="h-3 rounded-full bg-[#5B6CFF] transition-all"
            style={{ width: `${safeValue}%` }}
          />
        </div>
        <div className="mt-2 flex items-center justify-between">
          <span className="text-sm font-semibold text-slate-900">{safeValue}%</span>
          {helper && <span className="text-sm text-slate-500">{helper}</span>}
        </div>
      </div>
    )
  }