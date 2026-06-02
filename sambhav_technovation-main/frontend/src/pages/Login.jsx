import { useState } from "react"
import { useNavigate } from "react-router-dom"
import Logo from "../components/Logo"
import SamGuide from "../components/SamGuide"
import PrimaryButton from "../components/PrimaryButton"

export default function Login() {
  const navigate = useNavigate()
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")

  const handleLogin = () => {
    navigate("/dashboard")
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#dff7f2] via-[#e6f0ff] to-[#f5eaff] px-6 py-6">
      <div className="mx-auto flex min-h-screen max-w-5xl flex-col justify-between">
        <div className="pt-2">
          <Logo />
        </div>

        <div className="flex flex-1 items-center justify-center">
          <div className="w-full max-w-md rounded-3xl bg-white p-8 shadow-xl">
            <p className="text-sm font-semibold uppercase tracking-wide text-[#5B6CFF]">
              Welcome back
            </p>

            <h1 className="mt-2 text-3xl font-bold text-slate-900">
              Log in to Sambhav
            </h1>

            <p className="mt-3 text-slate-600">
              Use the dummy account for demos and submissions.
            </p>

            <div className="mt-6 space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium text-slate-700">
                  Username
                </label>
                <input
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="student"
                  className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-base outline-none transition focus:border-[#5B6CFF]"
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-slate-700">
                  Password
                </label>
                <input
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  type="password"
                  placeholder="1234"
                  className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-base outline-none transition focus:border-[#5B6CFF]"
                />
              </div>

              <PrimaryButton onClick={handleLogin} className="w-full py-4 text-lg">
                Continue
              </PrimaryButton>

              <button
                onClick={() => navigate("/")}
                className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-medium text-slate-600 transition hover:bg-slate-50"
              >
                Back to splash
              </button>
            </div>

            <div className="mt-6">
              <SamGuide text="I will guide you once you log in." />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}