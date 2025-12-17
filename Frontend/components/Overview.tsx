import type { OverviewProps } from "@/types"

export default function Overview({ results }: OverviewProps) {
  const totalRecords = results.length
  const nightmareCases = results.filter((r) => r.classification.is_nightmare).length
  const normalCases = totalRecords - nightmareCases

  const nightmarePercentage = totalRecords > 0 ? ((nightmareCases / totalRecords) * 100).toFixed(0) : 0
  const normalPercentage = totalRecords > 0 ? ((normalCases / totalRecords) * 100).toFixed(0) : 0

  const remCases = results.filter((r) => r.stage === "REM").length
  const n2Cases = results.filter((r) => r.stage === "N2").length

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-foreground mb-2">Session Overview</h2>
        <p className="text-base text-muted-foreground font-medium">Summary of all analyses in this session</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Total Records Card */}
        <div className="bg-card rounded-xl border border-border p-6 transition-smooth hover:shadow-md hover:border-accent/30">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Total Records</h3>
          <p className="text-5xl font-bold text-foreground mb-2">{totalRecords}</p>
          <div className="flex gap-2 text-xs mt-3">
            <span className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded font-medium">
              {remCases} REM
            </span>
            <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded font-medium">
              {n2Cases} N2
            </span>
          </div>
        </div>

        {/* Nightmare Cases Card */}
        <div className="bg-card rounded-xl border border-border p-6 transition-smooth hover:shadow-md hover:border-accent/30">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Nightmare Cases</h3>
          <p className="text-5xl font-bold text-red-600 dark:text-red-400 mb-2">{nightmareCases}</p>
          <p className="text-sm text-muted-foreground font-medium">{nightmarePercentage}% of total</p>
        </div>

        {/* Normal Cases Card */}
        <div className="bg-card rounded-xl border border-border p-6 transition-smooth hover:shadow-md hover:border-accent/30">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Normal Cases</h3>
          <p className="text-5xl font-bold text-green-600 dark:text-green-400 mb-2">{normalCases}</p>
          <p className="text-sm text-muted-foreground font-medium">{normalPercentage}% of total</p>
        </div>
      </div>

      {totalRecords === 0 && (
        <div className="bg-secondary border border-border rounded-xl p-8 text-center transition-smooth">
          <div className="w-16 h-16 rounded-full bg-accent/10 flex items-center justify-center mx-auto mb-4">
            <span className="text-3xl">📊</span>
          </div>
          <p className="text-muted-foreground font-medium">No data yet. Upload spectrograms to see analytics.</p>
        </div>
      )}

      {totalRecords > 0 && (
        <div className="bg-card rounded-xl border border-border p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4">Recent Activity</h3>
          <div className="space-y-3">
            {results
              .slice(-5)
              .reverse()
              .map((result, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-secondary rounded-lg border border-border/50"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`w-2 h-2 rounded-full ${result.classification.is_nightmare ? "bg-red-500" : "bg-green-500"}`}
                    />
                    <div>
                      <p className="text-sm font-medium text-foreground">
                        {result.filename || `Analysis ${results.length - index}`}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(result.timestamp).toLocaleTimeString()} • {result.stage} Stage
                      </p>
                    </div>
                  </div>
                  <span
                    className={`text-xs font-semibold px-2 py-1 rounded ${
                      result.classification.is_nightmare
                        ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"
                        : "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
                    }`}
                  >
                    {result.classification.severity}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}
