import { useRef, useState } from "react"

export default function SamBubble({ onAudioReady, disabled = false }) {
  const [isRecording, setIsRecording] = useState(false)
  const [level, setLevel] = useState(0)

  const mediaRecorderRef = useRef(null)
  const audioChunks = useRef([])
  const analyserRef = useRef(null)
  const animationRef = useRef(null)
  const streamRef = useRef(null)
  const audioContextRef = useRef(null)

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunks.current = []

      mediaRecorder.ondataavailable = (e) => {
        audioChunks.current.push(e.data)
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks.current, { type: "audio/webm" })
        if (onAudioReady) onAudioReady(blob)
      }

      mediaRecorder.start()

      const AudioCtx = window.AudioContext || window.webkitAudioContext
      const audioContext = new AudioCtx()
      audioContextRef.current = audioContext

      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 128
      source.connect(analyser)

      analyserRef.current = analyser
      setIsRecording(true)
      animate()
    } catch (err) {
      console.error("Mic error:", err)
    }
  }

  const stopRecording = () => {
    try {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop()
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
        streamRef.current = null
      }

      if (audioContextRef.current) {
        audioContextRef.current.close().catch(() => {})
        audioContextRef.current = null
      }

      setIsRecording(false)
      setLevel(0)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    } catch (err) {
      console.error(err)
    }
  }

  const handleClick = () => {
    if (disabled) return
    if (!isRecording) startRecording()
    else stopRecording()
  }

  const animate = () => {
    const analyser = analyserRef.current
    if (!analyser) return

    const dataArray = new Uint8Array(analyser.frequencyBinCount)

    const loop = () => {
      animationRef.current = requestAnimationFrame(loop)

      analyser.getByteFrequencyData(dataArray)
      const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length
      setLevel(avg / 255)
    }

    loop()
  }

  return (
    <div className="relative flex h-[320px] w-[320px] items-center justify-center">
      <div
        className="pointer-events-none absolute rounded-full bg-teal-400/15 blur-3xl transition-all duration-200"
        style={{
          width: 220 + level * 80,
          height: 220 + level * 80,
          transform: `scale(${1 + level * 0.15})`,
        }}
      />

      <button
        type="button"
        onClick={handleClick}
        disabled={disabled}
        className={`relative z-10 flex items-center justify-center rounded-full shadow-2xl transition-all duration-200 ${
          disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer"
        }`}
        style={{
          width: 148 + level * 12,
          height: 148 + level * 12,
          transform: `scale(${1 + level * 0.12})`,
          background: isRecording
            ? "linear-gradient(145deg, #ef4444, #dc2626)"
            : "linear-gradient(145deg, #14b8a6, #3b82f6)",
          boxShadow: isRecording
            ? "0 0 40px rgba(239,68,68,0.35)"
            : "0 0 40px rgba(20,184,166,0.25)",
        }}
      >
        <div
          className="flex items-center justify-center rounded-full text-white"
          style={{
            width: 88,
            height: 88,
            background: "rgba(255,255,255,0.18)",
            backdropFilter: "blur(10px)",
            fontSize: "1.9rem",
          }}
        >
          🎤
        </div>
      </button>

      <p className="absolute -bottom-12 text-lg font-medium text-slate-700">
        {disabled ? "Please wait..." : isRecording ? "Listening..." : "Tap to speak"}
      </p>
    </div>
  )
}