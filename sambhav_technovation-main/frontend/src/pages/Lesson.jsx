import { useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import Logo from "../components/Logo"
import SamGuide from "../components/SamGuide"
import SamBubble from "../components/SamBubble"
import PrimaryButton from "../components/PrimaryButton"
import { recordSession } from "../utils/storage"
import { analyzeAudio } from "../utils/api"

const words = ["cat", "bat", "dog", "sun", "cup"]
const MAX_ATTEMPTS = 4

export default function Lesson() {
  const navigate = useNavigate()

  const [index, setIndex] = useState(0)
  const [attempts, setAttempts] = useState(0)
  const [feedback, setFeedback] = useState("")
  const [errorPhoneme, setErrorPhoneme] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)

  const resultsRef = useRef([])
  const weakPhonemesRef = useRef({})

  const currentWord = words[index]

  const speak = (text) => {
    return new Promise((resolve) => {
      const synth = window.speechSynthesis
      synth.cancel()

      const utterance = new SpeechSynthesisUtterance(text)
      const voices = synth.getVoices()
      const voice =
        voices.find((v) => v.name.includes("Google") && v.lang === "en-US") ||
        voices.find((v) => v.lang === "en-US")

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
      k: "kuh",
      g: "guh",
      t: "tuh",
      d: "duh",
      r: "rr",
      l: "luh",
      th: "th",
      dh: "dh",
    }

    const sound = phonemeMap[errorPhoneme] || errorPhoneme || ""
    const sentence = errorPhoneme
      ? `${feedback}. Try saying ${sound} in ${currentWord}.`
      : feedback

    await speak(sentence)
  }

  const addWeakPhoneme = (phoneme) => {
    if (!phoneme) return
    weakPhonemesRef.current = {
      ...weakPhonemesRef.current,
      [phoneme]: (weakPhonemesRef.current[phoneme] || 0) + 1,
    }
  }

  const storeFinalResult = ({ correct, attemptsUsed, phoneme, skipped = false }) => {
    resultsRef.current = [
      ...resultsRef.current,
      {
        word: currentWord,
        correct,
        attempts: attemptsUsed,
        errorPhoneme: phoneme || null,
        skipped,
      },
    ]
  }

  const saveLessonReport = () => {
    const results = resultsRef.current
    const total = results.length
    const correctCount = results.filter((r) => r.correct).length
    const accuracy = total ? Math.round((correctCount / total) * 100) : 0

    const reportState = {
      diagnosis: false,
      practice: false,
      results,
      weakPhonemes: weakPhonemesRef.current,
      groupScores: {},
      accuracy,
      title: "Core Lesson",
    }

    recordSession({
      type: "lesson",
      title: "Core Lesson",
      accuracy,
      wordsPracticed: total,
      weakPhonemes: weakPhonemesRef.current,
      createdAt: new Date().toISOString(),
    })

    window.localStorage.setItem("sambhav_last_report", JSON.stringify(reportState))
    navigate("/report", { state: reportState })
  }

  const advance = () => {
    if (index === words.length - 1) {
      saveLessonReport()
      return
    }

    setIndex((prev) => prev + 1)
    setAttempts(0)
    setFeedback("")
    setErrorPhoneme(null)
  }

  const finalizeAndAdvance = ({ correct, attemptsUsed, phoneme, skipped = false }) => {
    storeFinalResult({ correct, attemptsUsed, phoneme, skipped })

    setTimeout(() => {
      if (index === words.length - 1) {
        saveLessonReport()
      } else {
        advance()
      }
    }, 900)
  }

  const handleAudio = async (audioBlob) => {
    const formData = new FormData()
    formData.append("file", audioBlob)
    formData.append("expected_word", currentWord)

    setIsProcessing(true)

    try {
      const data = await analyzeAudio(formData)
      const correct = Boolean(data.correct)
      const nextAttempt = attempts + 1

      let message = data.feedback || "Good try."
      if (!correct && data.error_phoneme) {
        message = `${message} Focus on the '${data.error_phoneme}' sound.`
      }

      setFeedback(message)
      setErrorPhoneme(data.error_phoneme || null)

      if (!correct && data.error_phoneme) {
        addWeakPhoneme(data.error_phoneme)
      }

      if (correct) {
        setAttempts(nextAttempt)
        finalizeAndAdvance({
          correct: true,
          attemptsUsed: nextAttempt,
          phoneme: data.error_phoneme || null,
        })
        return
      }

      if (nextAttempt >= MAX_ATTEMPTS) {
        setAttempts(nextAttempt)
        finalizeAndAdvance({
          correct: false,
          attemptsUsed: nextAttempt,
          phoneme: data.error_phoneme || null,
        })
        return
      }

      setAttempts(nextAttempt)
    } catch (err) {
      console.error(err)
      setFeedback("Error processing audio")
    } finally {
      setIsProcessing(false)
    }
  }

  const skipWord = () => {
    storeFinalResult({
      correct: false,
      attemptsUsed: attempts,
      phoneme: errorPhoneme,
      skipped: true,
    })

    setTimeout(() => {
      if (index === words.length - 1) {
        saveLessonReport()
      } else {
        advance()
      }
    }, 400)
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

  const progress = ((index + 1) / words.length) * 100

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#dff7f2] via-[#e6f0ff] to-[#f5eaff] px-6 py-6">
      <div className="mx-auto max-w-5xl">
        <Logo />

        <div className="mt-6 text-center">
          <p className="text-sm font-semibold uppercase tracking-wide text-[#5B6CFF]">
            Core learning
          </p>
          <h1 className="mt-2 text-4xl font-bold text-slate-900">
            {currentWord}
          </h1>
          <p className="mt-3 text-lg text-slate-600">
            Say the word clearly. Sam will help you step by step.
          </p>
        </div>

        <SamGuide
          className="mt-6"
          text={
            feedback
              ? feedback
              : "Tap the microphone and say the word when you are ready."
          }
        />

        <div className="mt-8 rounded-3xl bg-white p-6 shadow-xl">
          <div className="flex items-center justify-between gap-4">
            <p className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              Attempt {attempts} of {MAX_ATTEMPTS}
            </p>
            <p className="text-sm font-medium text-slate-500">
              Word {index + 1} of {words.length}
            </p>
          </div>

          <div className="mt-5 h-3 w-full rounded-full bg-slate-200">
            <div
              className="h-3 rounded-full bg-[#5B6CFF] transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="mt-8 flex justify-center">
            <SamBubble onAudioReady={handleAudio} disabled={isSpeaking} />
          </div>

          <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:justify-center">
            <button
              onClick={skipWord}
              className="rounded-2xl border border-slate-200 bg-white px-5 py-3 text-base font-medium text-slate-700 transition hover:bg-slate-50"
            >
              Skip
            </button>

            <PrimaryButton onClick={speakFeedback} disabled={!feedback || isSpeaking}>
              Hear Feedback
            </PrimaryButton>
          </div>

          {feedback && (
            <p className="mt-6 text-center text-lg text-slate-800">
              {feedback}
            </p>
          )}
        </div>

        <div className="mt-6">
          <SamGuide text="If the sound feels hard, you can skip it and come back later." />
        </div>
      </div>
    </div>
  )
}