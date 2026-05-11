import { usePolling } from "../hooks/usePolling";
import { fetchMaintenanceAlerts } from "../api/dashboardApi";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from "recharts";

const priorityLevel = (pFailure) => {
  if (pFailure >= 0.8) return "CRITICAL";
  if (pFailure >= 0.5) return "HIGH";
  return "MEDIUM";
};

export default function MaintenanceAlerts({ tenantId, wsAnomalies = [] }) {
  const { data: historical, loading, error } = usePolling(
    () => fetchMaintenanceAlerts(tenantId, 100),
    60000
  );

  // Live anomalies from WebSocket that have p_failure
  const liveAlerts = wsAnomalies.filter(r => r.p_failure != null);
  const data = liveAlerts.length > 0
    ? [...liveAlerts, ...(historical || [])].slice(0, 150)
    : (historical || []);

  if (loading && !liveAlerts.length) return <p className="status-msg">Loading alerts...</p>;
  if (error && !liveAlerts.length)   return <p className="status-msg error">{error}</p>;
  if (!data?.length) return <p className="status-msg">No maintenance alerts found.</p>;

  const sorted = [...data].sort(
    (a, b) => parseFloat(b.p_failure ?? 0) - parseFloat(a.p_failure ?? 0)
  );

  const scatterData = sorted.slice(0, 50).map((r) => ({
    x: parseFloat(r.anomalyScore ?? 0),
    y: parseFloat(r.p_failure ?? 0),
    name: r.segmentId,
  }));

  const critical = sorted.filter((r) => parseFloat(r.p_failure) >= 0.8).length;
  const high     = sorted.filter((r) => parseFloat(r.p_failure) >= 0.5 && parseFloat(r.p_failure) < 0.8).length;

  return (
    <div className="tab-content">
      <h2>Maintenance Alerts</h2>

      <div className="stat-row">
        <div className="stat-card critical"><span>{critical}</span><label>CRITICAL</label></div>
        <div className="stat-card high"><span>{high}</span><label>HIGH</label></div>
        <div className="stat-card total"><span>{sorted.length}</span><label>TOTAL ALERTS</label></div>
      </div>

      <div className="chart-box">
        <h3>Anomaly Score vs P(Failure)</h3>
        <ResponsiveContainer width="100%" height={250}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="x" name="Anomaly Score" tick={{ fill: "#aaa", fontSize: 11 }} label={{ value: "Anomaly Score", fill: "#aaa", position: "insideBottom", offset: -5 }} />
            <YAxis dataKey="y" name="P(Failure)" domain={[0, 1]} tick={{ fill: "#aaa", fontSize: 11 }} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ backgroundColor: "#1e1e1e", border: "1px solid #444" }} formatter={(v, n) => [v.toFixed(4), n]} />
            <Scatter data={scatterData} fill="#f87171" opacity={0.7} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="table-box">
        <table>
          <thead>
            <tr>
              <th>Segment</th><th>P(Failure)</th><th>Score</th>
              <th>Pressure</th><th>Flow</th><th>Vibration</th><th>Priority</th>
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 25).map((r, i) => {
              const pf = parseFloat(r.p_failure ?? 0);
              const level = priorityLevel(pf);
              return (
                <tr key={i}>
                  <td>{r.segmentId}</td>
                  <td className="score-cell">{pf.toFixed(6)}</td>
                  <td>{parseFloat(r.anomalyScore ?? 0).toFixed(4)}</td>
                  <td>{parseFloat(r.pressure ?? 0).toFixed(2)}</td>
                  <td>{parseFloat(r.flow ?? 0).toFixed(2)}</td>
                  <td>{parseFloat(r.vibration ?? 0).toFixed(2)}</td>
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
