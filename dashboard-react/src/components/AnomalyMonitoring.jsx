import { usePolling } from "../hooks/usePolling";
import { fetchAnomalies } from "../api/dashboardApi";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from "recharts";

export default function AnomalyMonitoring({ tenantId, wsAnomalies = [] }) {
  const { data: historical, loading, error } = usePolling(
    () => fetchAnomalies(tenantId, 100),
    60000
  );

  // Merge WebSocket live records with historical, newest first
  const data = wsAnomalies.length > 0
    ? [...wsAnomalies, ...(historical || [])].slice(0, 150)
    : (historical || []);

  if (loading && !wsAnomalies.length) return <p className="status-msg">Loading anomalies...</p>;
  if (error && !wsAnomalies.length)   return <p className="status-msg error">{error}</p>;
  if (!data?.length) return <p className="status-msg">No anomalies found.</p>;

  const chartData = [...data]
    .sort((a, b) => a.timestamp?.localeCompare(b.timestamp))
    .slice(-50)
    .map((r) => ({
      time: r.timestamp?.slice(11, 19) ?? "",
      pressure: parseFloat(r.pressure) || 0,
      flow: parseFloat(r.flow) || 0,
      vibration: parseFloat(r.vibration) || 0,
      score: parseFloat(r.anomalyScore) || 0,
    }));

  return (
    <div className="tab-content">
      <h2>Anomaly Monitoring</h2>

      <div className="chart-box">
        <h3>Sensor Readings — Anomalous Records</h3>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="time" tick={{ fill: "#aaa", fontSize: 11 }} />
            <YAxis tick={{ fill: "#aaa", fontSize: 11 }} />
            <Tooltip contentStyle={{ backgroundColor: "#1e1e1e", border: "1px solid #444" }} />
            <Legend />
            <Line type="monotone" dataKey="pressure" stroke="#38bdf8" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="flow" stroke="#34d399" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="vibration" stroke="#f59e0b" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-box">
        <h3>Anomaly Score Over Time</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="time" tick={{ fill: "#aaa", fontSize: 11 }} />
            <YAxis tick={{ fill: "#aaa", fontSize: 11 }} />
            <Tooltip contentStyle={{ backgroundColor: "#1e1e1e", border: "1px solid #444" }} />
            <Line type="monotone" dataKey="score" stroke="#f87171" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="table-box">
        <h3>Recent Anomalies ({data.length}) {wsAnomalies.length > 0 && <span className="live-badge">⚡ {wsAnomalies.length} live</span>}</h3>
        <table>
          <thead>
            <tr>
              <th>Segment</th><th>Timestamp</th>
              <th>Pressure</th><th>Flow</th><th>Vibration</th><th>Score</th>
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 20).map((r, i) => (
              <tr key={i}>
                <td>{r.segmentId}</td>
                <td>{r.timestamp?.slice(0, 19)}</td>
                <td>{parseFloat(r.pressure).toFixed(2)}</td>
                <td>{parseFloat(r.flow).toFixed(2)}</td>
                <td>{parseFloat(r.vibration).toFixed(2)}</td>
                <td className="score-cell">{parseFloat(r.anomalyScore).toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
