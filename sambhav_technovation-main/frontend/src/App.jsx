import { BrowserRouter, Routes, Route } from "react-router-dom"

import Splash from "./pages/Splash"
import Login from "./pages/Login"
import Dashboard from "./pages/Dashboard"
import Lesson from "./pages/Lesson"
import Report from "./pages/Report"
import About from "./pages/About"
import Diagnosis from "./pages/Diagnosis"
import PracticeHub from "./pages/PracticeHub"
import PracticeGroup from "./pages/PracticeGroup"
import PracticeLesson from "./pages/PracticeLesson"

export default function App() {
  return (
    <BrowserRouter basename="/voice-lab">
      <Routes>
        <Route path="/" element={<Splash />} />
        <Route path="/login" element={<Login />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/lesson" element={<Lesson />} />
        <Route path="/report" element={<Report />} />
        <Route path="/about" element={<About />} />
        <Route path="/diagnosis" element={<Diagnosis />} />
        <Route path="/practice" element={<PracticeHub />} />
        <Route path="/practice/:groupId" element={<PracticeGroup />} />
        <Route path="/practice/:groupId/:lessonId" element={<PracticeLesson />} />
      </Routes>
    </BrowserRouter>
  )
}