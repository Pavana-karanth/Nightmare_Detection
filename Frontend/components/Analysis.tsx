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

      <div className="space-y-6">
        {results.map((result, index) => (
          <div
            key={index}
            className="bg-card rounded-xl border border-border overflow-hidden transition-smooth hover:shadow-md hover:border-accent/30"
          >
            <div className="bg-secondary px-6 py-4 border-b border-border">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-foreground">{result.filename || `Analysis ${index + 1}`}</h3>
                  <p className="text-xs text-muted-foreground mt-1 font-medium">
                    {new Date(result.timestamp).toLocaleString()}
                  </p>
                </div>
                <span
                  className={`px-4 py-2 rounded-full text-sm font-semibold border ${getSeverityColor(result.classification.severity)}`}
                >
                  {getSeverityIcon(result.classification.severity)} {result.classification.severity.toUpperCase()}
                </span>
              </div>
            </div>

            <div className="p-6 space-y-6">
              {/* Classification Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-secondary rounded-lg p-4">
                  <p className="text-xs text-muted-foreground mb-2 font-semibold uppercase tracking-wide">
                    Nightmare Probability
                  </p>
                  <p className="text-3xl font-bold text-foreground">{result.classification.nightmare_probability}%</p>
                </div>
                <div className="bg-secondary rounded-lg p-4">
                  <p className="text-xs text-muted-foreground mb-2 font-semibold uppercase tracking-wide">
                    Normal Probability
                  </p>
                  <p className="text-3xl font-bold text-foreground">{result.classification.normal_probability}%</p>
                </div>
                <div className="bg-secondary rounded-lg p-4">
                  <p className="text-xs text-muted-foreground mb-2 font-semibold uppercase tracking-wide">Confidence</p>
                  <p className="text-3xl font-bold text-foreground">{result.classification.confidence}%</p>
                </div>
                <div className="bg-secondary rounded-lg p-4">
                  <p className="text-xs text-muted-foreground mb-2 font-semibold uppercase tracking-wide">
                    Anomaly Score
                  </p>
                  <p className="text-3xl font-bold text-foreground">{result.classification.anomaly_score}</p>
                </div>
              </div>

              {/* Progress Bar */}
              <div>
                <div className="flex justify-between text-xs text-muted-foreground mb-3 font-semibold">
                  <span>Normal</span>
                  <span>Nightmare</span>
                </div>
                <div className="h-3 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-green-500 to-red-500"
                    style={{ width: `${result.classification.nightmare_probability}%` }}
                  />
                </div>
              </div>

              {/* Insights */}
              <div>
                <h4 className="font-semibold text-foreground mb-3">Clinical Insights</h4>
                <ul className="space-y-2">
                  {result.insights.map((insight, i) => (
                    <li key={i} className="flex items-start space-x-2 text-sm text-muted-foreground font-medium">
                      <span className="text-accent mt-0.5">•</span>
                      <span>{insight}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Technical Details */}
              <details className="text-sm">
                <summary className="cursor-pointer font-semibold text-foreground hover:text-accent transition-smooth">
                  Technical Details
                </summary>
                <div className="mt-3 space-y-2 text-muted-foreground pl-4 font-medium">
                  <p>Model Version: {result.metadata.model_version}</p>
                  <p>Embedding Dimension: {result.metadata.embedding_dim}</p>
                  <p>File Type: {result.metadata.file_type}</p>
                  <p>Threshold: {result.classification.threshold}</p>
                  <p>Severity Level: {result.classification.severity_level}</p>
                </div>
              </details>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
