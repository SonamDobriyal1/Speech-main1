import { useRef, useState } from "react"
import { useNavigate, useParams } from "react-router-dom"
import Logo from "../components/Logo"
import SamGuide from "../components/SamGuide"
import SamBubble from "../components/SamBubble"
import PrimaryButton from "../components/PrimaryButton"
import { getPracticeGroupById, getPracticeLessonById } from "../data/practiceData"
import { recordSession } from "../utils/storage"
import { analyzeAudio } from "../utils/api"

const MAX_ATTEMPTS = 4

export default function PracticeLesson() {
  const { groupId, lessonId } = useParams()
  const navigate = useNavigate()

  const group = getPracticeGroupById(groupId)
  const lesson = getPracticeLessonById(groupId, lessonId)

  const [index, setIndex] = useState(0)
  const [attempts, setAttempts] = useState(0)
  const [feedback, setFeedback] = useState("")
  const [errorPhoneme, setErrorPhoneme] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)

  const resultsRef = useRef([])
  const weakPhonemesRef = useRef({})

  const currentPair = lesson?.words?.[index]

  // ---------------------------
  // TTS
  // ---------------------------
  const speak = (text) => {
    return new Promise((resolve) => {
      const synth = window.speechSynthesis
      synth.cancel()

      const utterance = new SpeechSynthesisUtterance(text)

      const voices = synth.getVoices()
      const voice =
        voices.find(v => v.name.includes("Google") && v.lang === "en-US") ||
        voices.find(v => v.lang === "en-US")

      if (voice) utterance.voice = voice

      utterance.rate = 0.85
      utterance.pitch = 1

      setIsSpeaking(true)

      utterance.onend = () => {
        setIsSpeaking(false)
        resolve()
      }

      utterance.onerror = () => {
        setIsSpeaking(false)
        resolve()
      }

      synth.speak(utterance)
    })
  }

  const speakFeedback = async () => {
    if (!feedback) return

    const phonemeMap = {
      b: "buh",
      p: "puh",
      r: "rr",
      l: "luh",
      k: "kuh",
      g: "guh",
      th: "th",
      dh: "dh",
      s: "suh",
      z: "zuh",
    }

    const sound = phonemeMap[errorPhoneme] || errorPhoneme || ""

    const sentence = errorPhoneme
      ? `${feedback}. Try saying ${sound} in ${currentPair.left}.`
      : feedback

    await speak(sentence)
  }

  // ---------------------------
  // Helpers
  // ---------------------------
  const addWeakPhoneme = (phoneme) => {
    if (!phoneme) return
    weakPhonemesRef.current = {
      ...weakPhonemesRef.current,
      [phoneme]: (weakPhonemesRef.current[phoneme] || 0) + 1,
    }
  }

  const saveSession = () => {
    const results = resultsRef.current
    const total = results.length
    const correct = results.filter(r => r.correct).length
    const accuracy = total ? Math.round((correct / total) * 100) : 0

    const reportState = {
      diagnosis: false,
      practice: true,
      title: "Targeted Practice",
      practiceGroup: group?.title,
      practiceLesson: lesson?.title,
      results,
      weakPhonemes: weakPhonemesRef.current,
      groupScores: {},
      accuracy,
    }

    recordSession({
      type: "practice",
      title: `${group?.title}: ${lesson?.title}`,
      accuracy,
      wordsPracticed: total,
      weakPhonemes: weakPhonemesRef.current,
      createdAt: new Date().toISOString(),
    })

    window.localStorage.setItem("sambhav_last_report", JSON.stringify(reportState))
    navigate("/report", { state: reportState })
  }

  const advance = () => {
    if (index === lesson.words.length - 1) {
      saveSession()
      return
    }

    setIndex(prev => prev + 1)
    setAttempts(0)
    setFeedback("")
    setErrorPhoneme(null)
  }

  // ---------------------------
  // AUDIO HANDLER
  // ---------------------------
  const handleAudio = async (audioBlob) => {
    if (!currentPair) return

    setIsProcessing(true)

    const formData = new FormData()
    formData.append("file", audioBlob)
    formData.append("expected_word", currentPair.left)
    formData.append("contrast_word", currentPair.right)

    try {
      const data = await analyzeAudio(formData)
      const correct = Boolean(data.correct)
      const nextAttempt = attempts + 1

      let message = data.feedback || "Good try."

      if (data.confused_with_pair) {
        message = `You said "${currentPair.right}" instead of "${currentPair.left}". Listen carefully to the difference.`
      }

      if (!correct && data.error_phoneme) {
        message = `${message} Focus on the '${data.error_phoneme}' sound.`
      }

      setFeedback(message)
      setErrorPhoneme(data.error_phoneme || null)

      if (!correct && data.error_phoneme) {
        addWeakPhoneme(data.error_phoneme)
      }

      resultsRef.current = [
        ...resultsRef.current,
        {
          target: currentPair.left,
          contrast: currentPair.right,
          correct,
          attempts: nextAttempt,
          errorPhoneme: data.error_phoneme || null,
        },
      ]

      if (correct || nextAttempt >= MAX_ATTEMPTS) {
        setAttempts(nextAttempt)
        setTimeout(() => advance(), 800)
      } else {
        setAttempts(nextAttempt)
      }

    } catch (err) {
      console.error(err)
      setFeedback("Error processing audio")
    } finally {
      setIsProcessing(false)
    }
  }

  const skipWord = () => {
    resultsRef.current = [
      ...resultsRef.current,
      {
        target: currentPair.left,
        contrast: currentPair.right,
        correct: false,
        attempts,
        errorPhoneme,
        skipped: true,
      },
    ]

    setTimeout(() => advance(), 300)
  }

  // ---------------------------
  // STATES
  // ---------------------------
  if (!group || !lesson || !currentPair) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p>Practice lesson not found.</p>
      </div>
    )
  }

  if (isProcessing) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-[#dff7f2] via-[#e6f0ff] to-[#f5eaff]">
        <div className="h-16 w-16 animate-spin rounded-full border-4 border-teal-400 border-t-transparent mb-6" />
        <p>Processing... please wait</p>
      </div>
    )
  }

  const progress = ((index + 1) / lesson.words.length) * 100

  // ---------------------------
  // UI
  // ---------------------------
  return (
    <div className="min-h-screen bg-gradient-to-br from-[#dff7f2] via-[#e6f0ff] to-[#f5eaff] px-6 py-6">
      <div className="max-w-5xl mx-auto">

        <Logo />

        <h1 className="text-4xl font-bold mt-6 text-center">
          {lesson.title}
        </h1>

        <SamGuide
          className="mt-6"
          text={
            feedback
              ? feedback
              : `Say "${currentPair.left}" clearly.`
          }
        />

        <div className="bg-white rounded-3xl p-6 mt-6 shadow-xl text-center">

          <p className="text-lg font-semibold text-slate-500">
            {currentPair.left}
          </p>

          <p className="text-sm text-slate-400 mt-1">
            (not {currentPair.right})
          </p>

          <div className="mt-6 flex justify-center">
            <SamBubble onAudioReady={handleAudio} disabled={isSpeaking} />
          </div>

          <p className="mt-4 text-sm text-slate-500">
            Attempt {attempts} of {MAX_ATTEMPTS}
          </p>

          <div className="mt-4 h-2 bg-gray-200 rounded-full">
            <div
              className="h-2 bg-[#5B6CFF] rounded-full"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="flex gap-3 justify-center mt-6">
            <button
              onClick={skipWord}
              className="px-4 py-2 bg-gray-200 rounded-xl"
            >
              Skip
            </button>

            <PrimaryButton
              onClick={speakFeedback}
              disabled={!feedback || isSpeaking}
            >
              Hear Feedback
            </PrimaryButton>
          </div>

          {feedback && (
            <p className="mt-4 text-lg">{feedback}</p>
          )}

        </div>

      </div>
    </div>
  )
}