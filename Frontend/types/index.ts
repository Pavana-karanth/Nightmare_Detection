// Shared TypeScript types for the application

export type Tab = 'overview' | 'upload' | 'analysis' | 'visualizations';

export interface Classification {
  is_nightmare: boolean;
  severity: 'normal' | 'mild' | 'moderate' | 'severe';
  severity_level: number;
  nightmare_probability: number;
  normal_probability: number;
  confidence: number;
  anomaly_score: number;
  threshold: number;
}

export interface Metadata {
  model_version: string;
  embedding_dim: number;
  file_type: string;
}

export interface AnalysisResult {
  status: string;
  timestamp: string;
  classification: Classification;
  insights: string[];
  metadata: Metadata;
  filename?: string;
}

export interface OverviewProps {
  results: AnalysisResult[];
}

export interface UploadProps {
  onAnalysisComplete: (result: AnalysisResult) => void;
}

export interface AnalysisProps {
  results: AnalysisResult[];
}

export interface VisualizationsProps {
  results: AnalysisResult[];
}

export interface HeaderProps {
  isDark: boolean;
  toggleDarkMode: () => void;
}

export interface TabNavigationProps {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
}