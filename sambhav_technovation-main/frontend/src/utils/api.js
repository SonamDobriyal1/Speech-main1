export async function analyzeAudio(formData) {
  const res = await fetch("/analyze", {
    method: "POST",
    body: formData,
  })

  if (!res.ok) {
    throw new Error("Analysis request failed")
  }

  return res.json()
}
