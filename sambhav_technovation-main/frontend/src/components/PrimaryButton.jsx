export default function PrimaryButton({
    children,
    onClick,
    className = "",
    disabled = false,
    type = "button",
  }) {
    return (
      <button
        type={type}
        onClick={onClick}
        disabled={disabled}
        className={`rounded-2xl px-5 py-3 text-base font-semibold transition ${
          disabled
            ? "cursor-not-allowed bg-slate-300 text-slate-500"
            : "bg-[#5B6CFF] text-white hover:bg-[#4a57f0]"
        } ${className}`}
      >
        {children}
      </button>
    )
  }