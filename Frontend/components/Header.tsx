"use client"

import type { HeaderProps } from "@/types"

export default function Header({ isDark, toggleDarkMode }: HeaderProps) {
  return (
    <header className="bg-card border-b border-border transition-smooth">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex justify-between items-start">
          <div className="flex-1">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent mb-2">
              NeuroTrack
            </h1>
            <p className="text-sm font-medium text-muted-foreground">EEG Spectrogram Analysis for Nightmare Disorder</p>
          </div>

          <button
            onClick={toggleDarkMode}
            className="px-4 py-2 rounded-lg border border-border bg-secondary hover:bg-border 
                     text-foreground transition-smooth flex items-center gap-2 font-medium"
          >
            <span className="text-lg">{isDark ? "☀️" : "🌙"}</span>
            <span>{isDark ? "Light" : "Dark"}</span>
          </button>
        </div>
      </div>
    </header>
  )
}
