"use client"

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
        {/* Classification Distribution */}
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
                    <div className="h-4 bg-secondary rounded-full overflow-hidden border border-border/50">
                      <div
                        className="h-full bg-gradient-to-r from-red-500 to-red-600 rounded-full transition-smooth shadow-sm"
                        style={{ width: `${nightmarePerc}%` }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground mt-1 font-medium">{nightmareCases} cases</p>
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-semibold text-foreground">Normal Cases</span>
                      <span className="text-sm font-bold text-green-600 dark:text-green-400">{normalPerc}%</span>
                    </div>
                    <div className="h-4 bg-secondary rounded-full overflow-hidden border border-border/50">
                      <div
                        className="h-full bg-gradient-to-r from-green-500 to-green-600 rounded-full transition-smooth shadow-sm"
                        style={{ width: `${normalPerc}%` }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground mt-1 font-medium">{normalCases} cases</p>
                  </div>
                </>
              )
            })()}
          </div>
        </div>

        <div className="bg-card rounded-xl border border-border p-6 transition-smooth hover:shadow-md hover:border-accent/30">
          <h3 className="text-lg font-semibold text-foreground mb-4">Sleep Stage Distribution</h3>
          <div className="space-y-4">
            {(() => {
              const remCases = results.filter((r) => r.stage === "REM").length
              const n2Cases = results.filter((r) => r.stage === "N2").length
              const remPerc = results.length > 0 ? ((remCases / results.length) * 100).toFixed(0) : 0
              const n2Perc = results.length > 0 ? ((n2Cases / results.length) * 100).toFixed(0) : 0

              return (
                <>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-semibold text-foreground">REM Stage</span>
                      <span className="text-sm font-bold text-purple-600 dark:text-purple-400">{remPerc}%</span>
                    </div>
                    <div className="h-4 bg-secondary rounded-full overflow-hidden border border-border/50">
                      <div
                        className="h-full bg-gradient-to-r from-purple-500 to-purple-600 rounded-full transition-smooth shadow-sm"
                        style={{ width: `${remPerc}%` }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground mt-1 font-medium">{remCases} cases</p>
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-semibold text-foreground">N2 Stage</span>
                      <span className="text-sm font-bold text-blue-600 dark:text-blue-400">{n2Perc}%</span>
                    </div>
                    <div className="h-4 bg-secondary rounded-full overflow-hidden border border-border/50">
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-full transition-smooth shadow-sm"
                        style={{ width: `${n2Perc}%` }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground mt-1 font-medium">{n2Cases} cases</p>
                  </div>
                </>
              )
            })()}
          </div>
        </div>

        {/* Confidence Statistics */}
        <div className="bg-card rounded-xl border border-border p-6 transition-smooth hover:shadow-md hover:border-accent/30">
          <h3 className="text-lg font-semibold text-foreground mb-4">Model Confidence</h3>
          <div className="space-y-4">
            {(() => {
              const avgConfidence = (
                results.reduce((sum, r) => sum + r.classification.confidence, 0) / results.length
              ).toFixed(1)
              const maxConfidence = Math.max(...results.map((r) => r.classification.confidence)).toFixed(1)
              const minConfidence = Math.min(...results.map((r) => r.classification.confidence)).toFixed(1)

              return (
                <>
                  <div className="text-center py-6 border-b border-border/50 bg-gradient-to-br from-accent/5 to-accent/10 rounded-lg">
                    <div className="text-5xl font-bold text-accent mb-2">{avgConfidence}%</div>
                    <p className="text-sm text-muted-foreground font-medium">Average Confidence</p>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-secondary rounded-lg p-4 border border-border/50">
                      <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wide mb-1">
                        Highest
                      </p>
                      <p className="text-2xl font-bold text-foreground">{maxConfidence}%</p>
                    </div>
                    <div className="bg-secondary rounded-lg p-4 border border-border/50">
                      <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wide mb-1">Lowest</p>
                      <p className="text-2xl font-bold text-foreground">{minConfidence}%</p>
                    </div>
                  </div>
                </>
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
                critical: 0,
              }
              results.forEach((r) => {
                severityMap[r.classification.severity]++
              })

              const severityColors: Record<string, string> = {
                normal: "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 border-green-300",
                nightmare: "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 border-red-300",
                // moderate: "bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 border-orange-300",
                // severe: "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 border-red-300",
                // critical: "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border-purple-300",
              }

              return Object.entries(severityMap).map(([severity, count]) => (
                <div
                  key={severity}
                  className={`p-4 rounded-lg ${severityColors[severity]} font-medium border-2 transition-smooth hover:scale-[1.02]`}
                >
                  <div className="flex justify-between items-center">
                    <span className="capitalize font-semibold text-sm">{severity}</span>
                    <span className="text-lg font-bold">
                      {count} case{count !== 1 ? "s" : ""}
                    </span>
                  </div>
                </div>
              ))
            })()}
          </div>
        </div>

        {/* Anomaly Analysis */}
        <div className="bg-card rounded-xl border border-border p-6 transition-smooth hover:shadow-md hover:border-accent/30 lg:col-span-2">
          <h3 className="text-lg font-semibold text-foreground mb-4">Anomaly Score Analysis</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {(() => {
              const avgAnomaly = (
                results.reduce((sum, r) => sum + r.classification.anomaly_score, 0) / results.length
              ).toFixed(3)
              const maxAnomaly = Math.max(...results.map((r) => r.classification.anomaly_score)).toFixed(3)
              const minAnomaly = Math.min(...results.map((r) => r.classification.anomaly_score)).toFixed(3)
              const avgThreshold = (
                results.reduce((sum, r) => sum + r.classification.radius_threshold, 0) / results.length
              ).toFixed(3)

              return (
                <>
                  <div className="bg-gradient-to-br from-accent/10 to-accent/5 rounded-lg p-4 border border-accent/30">
                    <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wide mb-2">
                      Avg Score
                    </p>
                    <p className="text-3xl font-bold text-accent">{avgAnomaly}</p>
                  </div>
                  <div className="bg-secondary rounded-lg p-4 border border-border/50">
                    <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wide mb-2">Highest</p>
                    <p className="text-3xl font-bold text-foreground">{maxAnomaly}</p>
                  </div>
                  <div className="bg-secondary rounded-lg p-4 border border-border/50">
                    <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wide mb-2">Lowest</p>
                    <p className="text-3xl font-bold text-foreground">{minAnomaly}</p>
                  </div>
                  <div className="bg-secondary rounded-lg p-4 border border-border/50">
                    <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wide mb-2">
                      Avg Threshold
                    </p>
                    <p className="text-3xl font-bold text-foreground">{avgThreshold}</p>
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
