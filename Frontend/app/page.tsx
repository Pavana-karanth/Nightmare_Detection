"use client"

import { useState, useEffect } from "react"
import Header from "@/components/Header"
import TabNavigation from "@/components/TabNavigation"
import Overview from "@/components/Overview"
import Upload from "@/components/Upload"
import Analysis from "@/components/Analysis"
import Visualizations from "@/components/Visualizations"
import type { Tab, AnalysisResult } from "@/types"

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("overview")
  const [results, setResults] = useState<AnalysisResult[]>([])
  const [isDark, setIsDark] = useState(false)

  useEffect(() => {
    const savedTheme = localStorage.getItem("theme")
    const prefersDark =
      savedTheme === "dark" || (!savedTheme && window.matchMedia("(prefers-color-scheme: dark)").matches)
    setIsDark(prefersDark)
    if (prefersDark) {
      document.documentElement.classList.add("dark")
    }
  }, [])

  const handleAnalysisComplete = (result: AnalysisResult) => {
    setResults((prev) => [...prev, result])
    setActiveTab("analysis")
  }

  const toggleDarkMode = () => {
    const newDarkMode = !isDark
    setIsDark(newDarkMode)
    document.documentElement.classList.toggle("dark")
    localStorage.setItem("theme", newDarkMode ? "dark" : "light")
  }

  return (
    <div className={`min-h-screen ${isDark ? "dark" : ""}`}>
      <div className="min-h-screen bg-background transition-smooth">
        <Header isDark={isDark} toggleDarkMode={toggleDarkMode} />

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

          <div className="mt-8">
            {activeTab === "overview" && <Overview results={results} />}
            {activeTab === "upload" && <Upload onAnalysisComplete={handleAnalysisComplete} />}
            {activeTab === "analysis" && <Analysis results={results} />}
            {activeTab === "visualizations" && <Visualizations results={results} />}
          </div>
        </main>
      </div>
    </div>
  )
}
