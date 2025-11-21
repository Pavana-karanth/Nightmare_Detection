import type { OverviewProps } from "@/types"

export default function Overview({ results }: OverviewProps) {
  const totalRecords = results.length
  const nightmareCases = results.filter((r) => r.classification.is_nightmare).length
  const normalCases = totalRecords - nightmareCases

  const nightmarePercentage = totalRecords > 0 ? ((nightmareCases / totalRecords) * 100).toFixed(0) : 0
  const normalPercentage = totalRecords > 0 ? ((normalCases / totalRecords) * 100).toFixed(0) : 0

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
          <p className="text-5xl font-bold text-foreground">{totalRecords}</p>
        </div>

        {/* Nightmare Cases Card */}
        <div className="bg-card rounded-xl border border-border p-6 transition-smooth hover:shadow-md hover:border-accent/30">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Nightmare Cases</h3>
          <p className="text-5xl font-bold text-red-600 dark:text-red-400">{nightmareCases}</p>
          <p className="text-sm text-muted-foreground mt-2 font-medium">{nightmarePercentage}% of total</p>
        </div>

        {/* Normal Cases Card */}
        <div className="bg-card rounded-xl border border-border p-6 transition-smooth hover:shadow-md hover:border-accent/30">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Normal Cases</h3>
          <p className="text-5xl font-bold text-green-600 dark:text-green-400">{normalCases}</p>
          <p className="text-sm text-muted-foreground mt-2 font-medium">{normalPercentage}% of total</p>
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
    </div>
  )
}
