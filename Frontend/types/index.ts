export type Tab = "overview" | "upload" | "analysis" | "visualizations"

export interface Classification {
  is_nightmare: boolean
  severity: "normal" | "mild" | "moderate" | "severe" | "critical"
  severity_level: number
  nightmare_probability: number
  normal_probability: number
  confidence: number
  anomaly_score: number
  radius_threshold: number
  score_vs_radius: number
}

export interface Visualization {
  image: string // base64-encoded PNG
  format: "base64_png"
}

export interface BandScore {
  raw_distance: number
  weight: number
  weighted_distance: number
}

export interface BandAnalysis {
  [bandName: string]: BandScore
}

export interface Metadata {
  model_version: string
  method: string
  channels: string[]
  band_weights: { [bandName: string]: number }
  embedding_dim: number
}

export interface AnalysisResult {
  status: string
  timestamp: string
  stage: "REM" | "N2"
  classification: Classification
  band_analysis: BandAnalysis
  insights: string[]
  visualization: Visualization
  metadata: Metadata
  filename?: string
}

export interface UploadProps {
  onAnalysisComplete: (result: AnalysisResult) => void
}

export interface AnalysisProps {
  results: AnalysisResult[]
}

export interface OverviewProps {
  results: AnalysisResult[]
}

export interface HeaderProps {
  isDark: boolean
  toggleDarkMode: () => void
}

export interface TabNavigationProps {
  activeTab: Tab
  onTabChange: (tab: Tab) => void
}
