"use client"

import type React from "react"

import { useState, useRef } from "react"
import type { UploadProps, AnalysisResult } from "@/types"

export default function Upload({ onAnalysisComplete }: UploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (file: File) => {
    const validTypes = ["image/png", "image/jpeg", "image/jpg", "application/octet-stream"]
    const validExtensions = [".png", ".jpg", ".jpeg", ".npy"]
    const fileExtension = "." + file.name.split(".").pop()?.toLowerCase()

    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
      setError("Invalid file type. Please upload PNG, JPG, JPEG, or NPY files.")
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
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Analysis failed")
      }

      const result: AnalysisResult = await response.json()
      result.filename = selectedFile.name
      onAnalysisComplete(result)
      setSelectedFile(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze spectrogram")
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="space-y-6">
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          border-2 border-dashed rounded-xl p-12 text-center transition-smooth
          ${isDragging ? "border-accent bg-accent/5" : "border-border bg-secondary"}
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
            <h3 className="text-lg font-bold text-foreground">Upload Spectrogram</h3>
            <p className="text-sm text-muted-foreground mt-1 font-medium">Drag and drop or click to select</p>
          </div>

          <p className="text-xs text-muted-foreground font-medium">PNG, JPG, JPEG, NPY (max 50MB)</p>

          <input
            ref={fileInputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.npy"
            onChange={handleFileInputChange}
            className="hidden"
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-6 py-3 bg-accent hover:bg-accent/90 text-accent-foreground font-semibold rounded-lg transition-smooth"
          >
            Select File
          </button>
        </div>
      </div>

      {selectedFile && (
        <div className="bg-card rounded-xl border border-border p-6 transition-smooth">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center">
                <span className="text-2xl">📊</span>
              </div>
              <div>
                <p className="font-semibold text-foreground">{selectedFile.name}</p>
                <p className="text-sm text-muted-foreground font-medium">{(selectedFile.size / 1024).toFixed(2)} KB</p>
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-muted 
                       text-white font-semibold rounded-lg transition-smooth disabled:cursor-not-allowed"
            >
              {isAnalyzing ? "Analyzing..." : "Analyze"}
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4 transition-smooth">
          <p className="text-red-700 dark:text-red-300 text-sm font-medium">{error}</p>
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
              <li>1. Upload an EEG spectrogram image or NumPy array</li>
              <li>2. Our Deep SVDD model analyzes the spectral patterns</li>
              <li>3. Get instant classification and severity assessment</li>
              <li>4. View detailed insights and recommendations</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  )
}
