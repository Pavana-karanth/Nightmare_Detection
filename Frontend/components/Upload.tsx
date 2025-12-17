"use client"

import type React from "react"
import { useState, useRef } from "react"
import axios from "axios"
import { API_ENDPOINTS } from "@/lib/config"
import type { UploadProps, AnalysisResult } from "@/types"

export default function Upload({ onAnalysisComplete }: UploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [sleepStage, setSleepStage] = useState<"REM" | "N2">("REM")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (file: File) => {
    const validExtensions = [".npy"]
    const fileExtension = "." + file.name.split(".").pop()?.toLowerCase()

    if (!validExtensions.includes(fileExtension)) {
      setError("Invalid file type. Please upload NPY files only.")
      return
    }

    if (file.size > 50 * 1024 * 1024) {
      setError("File size must be less than 50MB")
      return
    }

    setSelectedFile(file)
    setError(null)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return

    setIsAnalyzing(true)
    setError(null)

    const formData = new FormData()
    formData.append("file", selectedFile)

    try {
      const response = await axios.post<AnalysisResult>(`${API_ENDPOINTS.ANALYZE}?stage=${sleepStage}`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })

      const result = response.data
      result.filename = selectedFile.name
      onAnalysisComplete(result)
      setSelectedFile(null)
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const errorMessage = err.response?.data?.detail || err.message || "Analysis failed"
        setError(errorMessage)
      } else {
        setError("Failed to analyze spectrogram")
      }
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-card rounded-xl border border-border p-6">
        <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
          <span className="text-xl">🌙</span>
          Select Sleep Stage
        </h3>
        <div className="flex gap-4">
          <button
            onClick={() => setSleepStage("REM")}
            className={`flex-1 px-6 py-4 rounded-lg border-2 font-semibold transition-smooth ${
              sleepStage === "REM"
                ? "border-accent bg-accent text-accent-foreground shadow-lg"
                : "border-border bg-secondary text-muted-foreground hover:border-accent/50"
            }`}
          >
            <div className="text-center">
              <div className="text-2xl mb-1">💤</div>
              <div>REM Stage</div>
              <div className="text-xs mt-1 opacity-75">Rapid Eye Movement</div>
            </div>
          </button>
          <button
            onClick={() => setSleepStage("N2")}
            className={`flex-1 px-6 py-4 rounded-lg border-2 font-semibold transition-smooth ${
              sleepStage === "N2"
                ? "border-accent bg-accent text-accent-foreground shadow-lg"
                : "border-border bg-secondary text-muted-foreground hover:border-accent/50"
            }`}
          >
            <div className="text-center">
              <div className="text-2xl mb-1">😴</div>
              <div>N2 Stage</div>
              <div className="text-xs mt-1 opacity-75">Light Sleep</div>
            </div>
          </button>
        </div>
      </div>

      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          border-2 border-dashed rounded-xl p-12 text-center transition-smooth
          ${isDragging ? "border-accent bg-accent/5 scale-[1.02]" : "border-border bg-secondary"}
        `}
      >
        <div className="flex flex-col items-center space-y-4">
          <div className="w-16 h-16 bg-accent/10 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
          </div>

          <div>
            <h3 className="text-lg font-bold text-foreground">Upload EEG Spectrogram</h3>
            <p className="text-sm text-muted-foreground mt-1 font-medium">Drag and drop or click to select</p>
          </div>

          <p className="text-xs text-muted-foreground font-medium">NPY format only (max 50MB)</p>

          <input ref={fileInputRef} type="file" accept=".npy" onChange={handleFileInputChange} className="hidden" />

          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-6 py-3 bg-accent hover:bg-accent/90 text-accent-foreground font-semibold rounded-lg transition-smooth shadow-md hover:shadow-lg"
          >
            Select File
          </button>
        </div>
      </div>

      {selectedFile && (
        <div className="bg-card rounded-xl border border-border p-6 transition-smooth hover:shadow-md">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center">
                <span className="text-2xl">📊</span>
              </div>
              <div>
                <p className="font-semibold text-foreground">{selectedFile.name}</p>
                <p className="text-sm text-muted-foreground font-medium">
                  {(selectedFile.size / 1024).toFixed(2)} KB • {sleepStage} Stage
                </p>
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="px-6 py-3 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 
                       disabled:from-muted disabled:to-muted text-white font-semibold rounded-lg 
                       transition-smooth disabled:cursor-not-allowed shadow-md hover:shadow-lg"
            >
              {isAnalyzing ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                      fill="none"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  Analyzing...
                </span>
              ) : (
                "Analyze"
              )}
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4 transition-smooth">
          <div className="flex items-start gap-3">
            <span className="text-red-600 dark:text-red-400 text-xl">⚠️</span>
            <p className="text-red-700 dark:text-red-300 text-sm font-medium">{error}</p>
          </div>
        </div>
      )}

      <div className="bg-accent/5 border border-accent/20 rounded-xl p-6 transition-smooth">
        <div className="flex items-start space-x-3">
          <div className="w-6 h-6 bg-accent/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
            <span className="text-accent text-sm font-bold">i</span>
          </div>
          <div>
            <h4 className="font-semibold text-foreground mb-3">How it works</h4>
            <ol className="space-y-2 text-sm text-muted-foreground font-medium">
              <li>1. Select your sleep stage (REM or N2)</li>
              <li>2. Upload an EEG spectrogram in NPY format (4 channels, 100 freq bins)</li>
              <li>3. Our Band-Weighted Deep SVDD model analyzes spectral patterns</li>
              <li>4. Get instant classification with band-specific anomaly breakdown</li>
              <li>5. View clinical insights and severity assessment</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  )
}
