"use client"

import type { Tab, TabNavigationProps } from "@/types"

export default function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "overview", label: "Overview" },
    { id: "upload", label: "Upload" },
    { id: "analysis", label: "Analysis" },
    { id: "visualizations", label: "Visualizations" },
  ]

  return (
    <nav className="flex gap-2 bg-card border border-border rounded-xl p-2">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`
            flex-1 px-4 py-3 rounded-lg text-sm font-semibold transition-smooth
            ${
              activeTab === tab.id
                ? "bg-primary text-primary-foreground shadow-md"
                : "text-muted-foreground hover:text-foreground"
            }
          `}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  )
}
