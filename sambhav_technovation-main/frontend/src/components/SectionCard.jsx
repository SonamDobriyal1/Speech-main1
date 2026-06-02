export default function SectionCard({ title, subtitle, children, className = "" }) {
    return (
      <div className={`rounded-3xl bg-white p-6 shadow-lg ${className}`}>
        {title && <h2 className="text-2xl font-bold text-slate-900">{title}</h2>}
        {subtitle && <p className="mt-2 text-slate-600">{subtitle}</p>}
        {children}
      </div>
    )
  }