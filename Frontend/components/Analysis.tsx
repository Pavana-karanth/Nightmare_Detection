"use client"

import type { AnalysisProps } from "@/types"

export default function Analysis({ results }: AnalysisProps) {
  if (results.length === 0) {
    return (
      <div className="bg-card rounded-xl border border-border p-12 text-center transition-smooth">
        <div className="w-16 h-16 bg-secondary rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-3xl">📊</span>
        </div>
        <h3 className="text-lg font-semibold text-foreground mb-2">No Analysis Results Yet</h3>
        <p className="text-muted-foreground">Upload spectrograms to see detailed analysis results here.</p>
      </div>
    )
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "normal":
        return "bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 border-green-200 dark:border-green-800"
      case "mild":
        return "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300 border-yellow-200 dark:border-yellow-800"
      case "moderate":
        return "bg-orange-100 dark:bg-orange-900/30 text-orange-800 dark:text-orange-300 border-orange-200 dark:border-orange-800"
      case "severe":
        return "bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300 border-red-200 dark:border-red-800"
      case "critical":
        return "bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-300 border-purple-200 dark:border-purple-800"
      default:
        return "bg-secondary text-muted-foreground border-border"
    }
  }

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case "normal":
        return "✅"
      case "mild":
        return "⚠️"
      case "moderate":
        return "⚠️"
      case "severe":
        return "🔴"
      case "critical":
        return "🚨"
      default:
        return "•"
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-foreground mb-1">Analysis Results</h2>
        <p className="text-muted-foreground font-medium">
          {results.length} {results.length === 1 ? "result" : "results"} found
        </p>
      </div>

      <div className="space-y-8">
        {results.map((result, index) => (
          <div
            key={index}
            className="bg-card rounded-xl border border-border overflow-hidden transition-smooth hover:shadow-lg hover:border-accent/30"
          >
            {/* Header */}
            <div className="bg-gradient-to-r from-secondary to-secondary/50 px-6 py-4 border-b border-border">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div>
                  <h3 className="font-semibold text-foreground text-lg">
                    {result.filename || `Analysis ${index + 1}`}
                  </h3>
                  <p className="text-xs text-muted-foreground mt-1 font-medium">
                    {new Date(result.timestamp).toLocaleString()} • Stage: {result.stage}
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <span className="px-3 py-1.5 rounded-full text-xs font-bold bg-accent/10 text-accent border border-accent/30">
                    {result.stage} Stage
                  </span>
                  <span
                    className={`px-4 py-2 rounded-full text-sm font-semibold border ${getSeverityColor(result.classification.severity)}`}
                  >
                    {getSeverityIcon(result.classification.severity)} {result.classification.severity.toUpperCase()}
                  </span>
                </div>
              </div>
            </div>

            <div className="p-6 space-y-6">
              {result.visualization?.image && (
                <div className="rounded-lg overflow-hidden border-2 border-border bg-black/5 shadow-inner">
                  <img
                    src={`data:image/png;base64,${result.visualization.image}`}
                    alt="EEG Spectrogram Visualization"
                    className="w-full h-auto"
                  />
                  <div className="bg-secondary px-4 py-2 border-t border-border">
                    <p className="text-xs text-muted-foreground font-medium text-center">
                      EEG Spectrogram • {result.metadata.channels.join(", ")} Channels
                    </p>
                  </div>
                </div>
              )}

              {/* Classification Metrics Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 rounded-lg p-4 border border-red-200 dark:border-red-800/50">
                  <p className="text-xs text-red-700 dark:text-red-300 mb-2 font-semibold uppercase tracking-wide">
                    Nightmare Prob.
                  </p>
                  <p className="text-3xl font-bold text-red-600 dark:text-red-400">
                    {result.classification.nightmare_probability}%
                  </p>
                </div>
                <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg p-4 border border-green-200 dark:border-green-800/50">
                  <p className="text-xs text-green-700 dark:text-green-300 mb-2 font-semibold uppercase tracking-wide">
                    Normal Prob.
                  </p>
                  <p className="text-3xl font-bold text-green-600 dark:text-green-400">
                    {result.classification.normal_probability}%
                  </p>
                </div>
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800/50">
                  <p className="text-xs text-blue-700 dark:text-blue-300 mb-2 font-semibold uppercase tracking-wide">
                    Confidence
                  </p>
                  <p className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                    {result.classification.confidence}%
                  </p>
                </div>
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg p-4 border border-purple-200 dark:border-purple-800/50">
                  <p className="text-xs text-purple-700 dark:text-purple-300 mb-2 font-semibold uppercase tracking-wide">
                    Anomaly Score
                  </p>
                  <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                    {result.classification.anomaly_score.toFixed(3)}
                  </p>
                  <p className="text-xs text-purple-600/70 dark:text-purple-400/70 mt-1">
                    vs {result.classification.radius_threshold.toFixed(3)}
                  </p>
                </div>
              </div>

              {/* Severity Distribution Bar */}
              <div>
                <div className="flex justify-between text-xs text-muted-foreground mb-3 font-semibold">
                  <span>Normal Dreams</span>
                  <span>Nightmare Activity</span>
                </div>
                <div className="h-6 bg-secondary rounded-full overflow-hidden border-2 border-border/50 shadow-inner">
                  <div
                    className="h-full bg-gradient-to-r from-green-500 via-yellow-500 via-orange-500 to-red-500 transition-all duration-500 shadow-sm"
                    style={{ width: `${result.classification.nightmare_probability}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-muted-foreground mt-2 font-medium">
                  <span>{result.classification.normal_probability}%</span>
                  <span>{result.classification.nightmare_probability}%</span>
                </div>
              </div>

              <div className="bg-gradient-to-br from-accent/5 to-accent/10 border border-accent/30 rounded-lg p-5">
                <h4 className="font-semibold text-foreground mb-4 text-sm uppercase tracking-wide flex items-center gap-2">
                  <span className="text-lg">🔬</span>
                  Frequency Band Analysis
                </h4>
                <div className="space-y-3">
                  {Object.entries(result.band_analysis)
                    .sort((a, b) => b[1].weighted_distance - a[1].weighted_distance)
                    .map(([bandName, bandData]) => (
                      <div key={bandName} className="bg-card/50 rounded-lg p-3 border border-border/30">
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-semibold text-foreground capitalize text-sm">{bandName}</span>
                          <span className="text-xs font-bold text-accent">
                            Weight: {(bandData.weight * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <p className="text-muted-foreground">Raw Distance</p>
                            <p className="font-bold text-foreground">{bandData.raw_distance.toFixed(3)}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Weighted</p>
                            <p className="font-bold text-foreground">{bandData.weighted_distance.toFixed(3)}</p>
                          </div>
                        </div>
                        <div className="mt-2 h-2 bg-secondary rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-accent to-accent/60"
                            style={{
                              width: `${Math.min((bandData.weighted_distance / result.classification.anomaly_score) * 100, 100)}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>

              {/* Clinical Insights */}
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800/50 rounded-lg p-5">
                <h4 className="font-semibold text-foreground mb-4 text-sm uppercase tracking-wide flex items-center gap-2">
                  <span className="text-lg">💡</span>
                  Clinical Insights
                </h4>
                <ul className="space-y-3">
                  {result.insights.map((insight, i) => (
                    <li key={i} className="flex items-start space-x-3 text-sm text-foreground/90 font-medium">
                      <span className="text-blue-600 dark:text-blue-400 font-bold mt-0.5 text-base">•</span>
                      <span>{insight}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Technical Details */}
              <details className="text-sm">
                <summary className="cursor-pointer font-semibold text-foreground hover:text-accent transition-smooth py-2 flex items-center gap-2">
                  <span className="text-base">⚙️</span>
                  Technical Details
                </summary>
                <div className="mt-4 space-y-3 text-muted-foreground pl-4 font-medium bg-secondary/50 rounded-lg p-4 border border-border/50">
                  <div className="flex justify-between">
                    <span>Model Version:</span>
                    <span className="text-foreground font-semibold">{result.metadata.model_version}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Method:</span>
                    <span className="text-foreground font-semibold">{result.metadata.method}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Embedding Dimension:</span>
                    <span className="text-foreground font-semibold">{result.metadata.embedding_dim}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Channels:</span>
                    <span className="text-foreground font-semibold">{result.metadata.channels.join(", ")}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Radius Threshold:</span>
                    <span className="text-foreground font-semibold">
                      {result.classification.radius_threshold.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Score vs Radius:</span>
                    <span className="text-foreground font-semibold">
                      {result.classification.score_vs_radius.toFixed(2)}x
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Severity Level:</span>
                    <span className="text-foreground font-semibold">{result.classification.severity_level} / 4</span>
                  </div>
                </div>
              </details>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
