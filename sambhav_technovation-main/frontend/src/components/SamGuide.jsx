import sam from "../assets/sam.png"

export default function SamGuide({ text, className = "" }) {
  return (
    <div className={`flex items-end gap-3 ${className}`}>
      <img src={sam} alt="Sam" className="h-16 w-16 shrink-0" />
      <div className="max-w-md rounded-3xl bg-white px-4 py-3 text-sm text-slate-700 shadow-md">
        {text}
      </div>
    </div>
  )
}