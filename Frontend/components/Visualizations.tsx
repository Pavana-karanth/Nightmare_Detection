import type { AnalysisProps } from "@/types"

export default function Visualizations({ results }: AnalysisProps) {
  if (results.length === 0) {
    return (
      <div className="bg-card rounded-xl border border-border p-12 text-center transition-smooth">
        <div className="w-16 h-16 bg-secondary rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-3xl">📈</span>
        </div>
        <h3 className="text-lg font-semibold text-foreground mb-2">No Visualizations Available</h3>
        <p className="text-muted-foreground font-medium">Complete some analyses to see visualizations of your data.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-foreground mb-1">Data Visualizations</h2>
        <p className="text-muted-foreground font-medium">
          Graphical analysis of {results.length} {results.length === 1 ? "result" : "results"}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Distribution Overview */}
        <div className="bg-card rounded-xl border border-border p-6 transition-smooth hover:shadow-md hover:border-accent/30">
          <h3 className="text-lg font-semibold text-foreground mb-4">Classification Distribution</h3>
          <div className="space-y-4">
            {(() => {
              const nightmareCases = results.filter((r) => r.classification.is_nightmare).length
              const normalCases = results.length - nightmareCases
              const nightmarePerc = ((nightmareCases / results.length) * 100).toFixed(0)
              const normalPerc = ((normalCases / results.length) * 100).toFixed(0)

              return (
                <>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-semibold text-foreground">Nightmare Cases</span>
                      <span className="text-sm font-bold text-red-600 dark:text-red-400">{nightmarePerc}%</span>
                    </div>
                    <div className="h-3 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-red-500 rounded-full transition-smooth"
                        style={{ width: `${nightmarePerc}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-semibold text-foreground">Normal Cases</span>
                      <span className="text-sm font-bold text-green-600 dark:text-green-400">{normalPerc}%</span>
                    </div>
                    <div className="h-3 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-green-500 rounded-full transition-smooth"
                        style={{ width: `${normalPerc}%` }}
                      />
                    </div>
                  </div>
                </>
              )
            })()}
          </div>
        </div>

        {/* Confidence Statistics */}
        <div className="bg-card rounded-xl border border-border p-6 transition-smooth hover:shadow-md hover:border-accent/30">
          <h3 className="text-lg font-semibold text-foreground mb-4">Average Confidence</h3>
          <div className="space-y-4">
            {(() => {
              const avgConfidence = (
                results.reduce((sum, r) => sum + r.classification.confidence, 0) / results.length
              ).toFixed(1)

              return (
                <div className="text-center py-8">
                  <div className="text-5xl font-bold text-accent mb-2">{avgConfidence}%</div>
                  <p className="text-sm text-muted-foreground font-medium">Average Model Confidence</p>
                </div>
              )
            })()}
          </div>
        </div>

        {/* Severity Breakdown */}
        <div className="bg-card rounded-xl border border-border p-6 transition-smooth hover:shadow-md hover:border-accent/30">
          <h3 className="text-lg font-semibold text-foreground mb-4">Severity Breakdown</h3>
          <div className="space-y-3">
            {(() => {
              const severityMap: Record<string, number> = {
                normal: 0,
                mild: 0,
                moderate: 0,
                severe: 0,
              }
              results.forEach((r) => {
                severityMap[r.classification.severity]++
              })

              const severityColors: Record<string, string> = {
                normal: "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300",
                mild: "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300",
                moderate: "bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300",
                severe: "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300",
              }

              return Object.entries(severityMap).map(([severity, count]) => (
                <div key={severity} className={`p-3 rounded-lg ${severityColors[severity]} font-medium`}>
                  <div className="flex justify-between">
                    <span className="capitalize">{severity}</span>
                    <span className="font-bold">
                      {count} case{count !== 1 ? "s" : ""}
                    </span>
                  </div>
                </div>
              ))
            })()}
          </div>
        </div>

        {/* Anomaly Statistics */}
        <div className="bg-card rounded-xl border border-border p-6 transition-smooth hover:shadow-md hover:border-accent/30">
          <h3 className="text-lg font-semibold text-foreground mb-4">Anomaly Analysis</h3>
          <div className="space-y-4">
            {(() => {
              const avgAnomaly = (
                results.reduce((sum, r) => sum + r.classification.anomaly_score, 0) / results.length
              ).toFixed(2)
              const maxAnomaly = Math.max(...results.map((r) => r.classification.anomaly_score)).toFixed(2)
              const minAnomaly = Math.min(...results.map((r) => r.classification.anomaly_score)).toFixed(2)

              return (
                <>
                  <div>
                    <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wide mb-1">
                      Average Anomaly Score
                    </p>
                    <p className="text-3xl font-bold text-foreground">{avgAnomaly}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-secondary rounded-lg p-3">
                      <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wide mb-1">
                        Highest
                      </p>
                      <p className="text-xl font-bold text-foreground">{maxAnomaly}</p>
                    </div>
                    <div className="bg-secondary rounded-lg p-3">
                      <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wide mb-1">Lowest</p>
                      <p className="text-xl font-bold text-foreground">{minAnomaly}</p>
                    </div>
                  </div>
                </>
              )
            })()}
          </div>
        </div>
      </div>
    </div>
  )
}
