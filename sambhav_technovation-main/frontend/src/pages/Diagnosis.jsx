import { useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import Logo from "../components/Logo"
import SamGuide from "../components/SamGuide"
import SamBubble from "../components/SamBubble"
import { recordSession } from "../utils/storage"
import { analyzeAudio } from "../utils/api"

const diagnosisItems = [
  { word: "bat", groupId: "bp", contrastWord: "pat" },
  { word: "pat", groupId: "bp", contrastWord: "bat" },
  { word: "right", groupId: "rl", contrastWord: "light" },
  { word: "light", groupId: "rl", contrastWord: "right" },
  { word: "think", groupId: "th", contrastWord: "sink" },
  { word: "this", groupId: "th", contrastWord: "dis" },
  { word: "coat", groupId: "kg", contrastWord: "goat" },
  { word: "goat", groupId: "kg", contrastWord: "coat" },
]

export default function Diagnosis() {
  const navigate = useNavigate()

  const [index, setIndex] = useState(0)
  const [feedback, setFeedback] = useState("")
  const [isProcessing, setIsProcessing] = useState(false)

  const resultsRef = useRef([])
  const groupScoresRef = useRef({})
  const weakPhonemesRef = useRef({})

  const current = diagnosisItems[index]

  const updateGroupScore = (groupId, correct) => {
    const previous = groupScoresRef.current[groupId] || { correct: 0, total: 0 }
    groupScoresRef.current = {
      ...groupScoresRef.current,
      [groupId]: {
        correct: previous.correct + (correct ? 1 : 0),
        total: previous.total + 1,
      },
    }
  }

  const addWeakPhoneme = (phoneme) => {
    if (!phoneme) return
    weakPhonemesRef.current = {
      ...weakPhonemesRef.current,
      [phoneme]: (weakPhonemesRef.current[phoneme] || 0) + 1,
    }
  }

  const saveDiagnosisReport = () => {
    const results = resultsRef.current
    const total = results.length
    const correctCount = results.filter((r) => r.correct).length
    const accuracy = total ? Math.round((correctCount / total) * 100) : 0

    const reportState = {
      diagnosis: true,
      practice: false,
      title: "Screening Report",
      results,
      weakPhonemes: weakPhonemesRef.current,
      groupScores: groupScoresRef.current,
      accuracy,
    }

    recordSession({
      type: "diagnosis",
      title: "Screening Mode",
      accuracy,
      wordsPracticed: total,
      weakPhonemes: weakPhonemesRef.current,
      createdAt: new Date().toISOString(),
    })

    window.localStorage.setItem("sambhav_last_report", JSON.stringify(reportState))
    navigate("/report", { state: reportState })
  }

  const moveNext = () => {
    if (index === diagnosisItems.length - 1) {
      saveDiagnosisReport()
      return
    }

    setIndex((prev) => prev + 1)
    setFeedback("")
  }

  const handleAudio = async (audioBlob) => {
    if (!current) return

    setIsProcessing(true)

    const formData = new FormData()
    formData.append("file", audioBlob)
    formData.append("expected_word", current.word)
    formData.append("contrast_word", current.contrastWord)

    try {
      const data = await analyzeAudio(formData)
      const correct = Boolean(data.correct)

      const message = data.confused_with_pair
        ? `You said "${current.contrastWord}" instead of "${current.word}".`
        : data.feedback || "Good try."

      setFeedback(message)

      resultsRef.current = [
        ...resultsRef.current,
        {
          word: current.word,
          contrastWord: current.contrastWord,
          groupId: current.groupId,
          correct,
          errorPhoneme: data.error_phoneme || null,
        },
      ]

      updateGroupScore(current.groupId, correct)

      if (!correct && data.error_phoneme) {
        addWeakPhoneme(data.error_phoneme)
      }

      setTimeout(() => moveNext(), 900)
    } catch (err) {
      console.error(err)
      setFeedback("Error processing audio")
    } finally {
      setIsProcessing(false)
    }
  }

  if (isProcessing) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-[#dff7f2] via-[#e6f0ff] to-[#f5eaff] px-6">
        <div className="mb-6 h-16 w-16 animate-spin rounded-full border-4 border-teal-400 border-t-transparent" />
        <p className="text-lg font-medium text-slate-700">
          Processing... please wait
        </p>
      </div>
    )
  }

  if (!current) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-white">
        <p className="text-slate-700">Screening not found.</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#dff7f2] via-[#e6f0ff] to-[#f5eaff] px-6 py-6">
      <div className="mx-auto max-w-5xl">
        <Logo />

        <div className="mt-6 text-center">
          <p className="text-sm font-semibold uppercase tracking-wide text-[#7c3aed]">
            Screening mode
          </p>
          <h1 className="mt-2 text-4xl font-bold text-slate-900">
            Say the word clearly
          </h1>
          <p className="mt-3 text-lg text-slate-600">
            One item at a time. Sam will help show which sounds need more support.
          </p>
        </div>

        <SamGuide
          className="mt-6"
          text="Say each word once. I will save the results locally and help you review the patterns."
        />

        <div className="mt-8 rounded-3xl bg-white p-6 shadow-xl">
          <div className="flex items-center justify-between gap-4">
            <p className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              Item {index + 1} of {diagnosisItems.length}
            </p>
            <p className="text-sm font-medium text-slate-500">
              One pass per item
            </p>
          </div>

          <div className="mt-6 text-center">
            <p className="text-sm font-semibold uppercase tracking-wide text-[#5B6CFF]">
              Target word
            </p>
            <h2 className="mt-2 text-5xl font-bold text-slate-900">
              {current.word}
            </h2>
            <p className="mt-3 text-slate-600">
              Contrast word: {current.contrastWord}
            </p>
          </div>

          <div className="mt-8 flex justify-center">
            <SamBubble onAudioReady={handleAudio} />
          </div>

          {feedback && (
            <p className="mt-6 text-center text-lg text-slate-800">
              {feedback}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}