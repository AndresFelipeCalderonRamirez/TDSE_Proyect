import { usePolling } from "../hooks/usePolling";
import { fetchRanking } from "../api/dashboardApi";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from "recharts";

const riskColor = (score) => {
  if (score >= 0.8) return "#f87171";
  if (score >= 0.5) return "#f59e0b";
  return "#34d399";
};

export default function NetworkRisk({ tenantId }) {
  const { data, loading, error, lastUpdated } = usePolling(
    () => fetchRanking(tenantId),
    15000
  );

  if (loading) return <p className="status-msg">Loading ranking...</p>;
  if (error)   return <p className="status-msg error">{error}</p>;
  if (!data)   return <p className="status-msg">No ranking available yet.</p>;

  const segments = (data.segments || [])
    .sort((a, b) => (b.risk_propagated ?? 0) - (a.risk_propagated ?? 0))
    .slice(0, 10);

  const chartData = segments.map((s) => ({
    name: s.segment_id,
    risk: parseFloat(s.risk_propagated ?? 0).toFixed(4),
    failure: parseFloat(s.p_failure ?? 0).toFixed(4),
  }));

  return (
    <div className="tab-content">
      <h2>Network Risk — Digital Twin</h2>
      {lastUpdated && (
        <p className="last-updated">Last updated: {lastUpdated.toLocaleTimeString()}</p>
      )}

      <div className="chart-box">
        <h3>Top 10 Segments by Propagated Risk (α = 0.7)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis type="number" domain={[0, 1]} tick={{ fill: "#aaa", fontSize: 11 }} />
            <YAxis dataKey="name" type="category" tick={{ fill: "#aaa", fontSize: 11 }} width={90} />
            <Tooltip contentStyle={{ backgroundColor: "#1e1e1e", border: "1px solid #444" }} />
            <Bar dataKey="risk" name="Risk Score">
              {chartData.map((entry, i) => (
                <Cell key={i} fill={riskColor(parseFloat(entry.risk))} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="table-box">
        <h3>Ranking Detail</h3>
        <table>
          <thead>
            <tr>
              <th>Rank</th><th>Segment</th>
              <th>Risk Score</th><th>P(Failure)</th><th>Level</th>
            </tr>
          </thead>
          <tbody>
            {segments.map((s, i) => {
              const risk = parseFloat(s.risk_propagated ?? 0);
              const level = risk >= 0.8 ? "CRITICAL" : risk >= 0.5 ? "HIGH" : "MEDIUM";
              return (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>{s.segment_id}</td>
                  <td>{risk.toFixed(4)}</td>
                  <td>{parseFloat(s.p_failure ?? 0) > 0 ? parseFloat(s.p_failure).toFixed(4) : "—"}</td>
                  <td><span className={`badge badge-${level.toLowerCase()}`}>{level}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
