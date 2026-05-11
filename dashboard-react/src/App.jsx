import { useState } from "react";
import { usePolling } from "./hooks/usePolling";
import { useWebSocket } from "./hooks/useWebSocket";
import { fetchStats } from "./api/dashboardApi";
import AnomalyMonitoring from "./components/AnomalyMonitoring";
import NetworkRisk from "./components/NetworkRisk";
import MaintenanceAlerts from "./components/MaintenanceAlerts";
import QAMetrics from "./components/QAMetrics";
import "./App.css";

const TENANTS = ["tenant-A", "tenant-B"];
const TABS = ["Anomaly Monitoring", "Network Risk", "Maintenance Alerts", "QA Metrics"];

export default function App() {
  const [tenant, setTenant] = useState("tenant-A");
  const [activeTab, setActiveTab] = useState(0);

  const { data: stats, lastUpdated } = usePolling(
    () => fetchStats(tenant),
    30000
  );

  const { anomalies: wsAnomalies, connected: wsConnected } = useWebSocket(tenant);

  const liveAnomalyCount = (stats?.anomalies_detected ?? 0) + wsAnomalies.length;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-logo">💧</div>
        <h1 className="sidebar-title">WaterTwinML</h1>
        <p className="sidebar-sub">Operational Dashboard</p>

        <div className="sidebar-section">
          <label>Tenant</label>
          <select value={tenant} onChange={(e) => setTenant(e.target.value)}>
            {TENANTS.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        <div className="sidebar-section">
          <div className="conn-indicator">● DynamoDB Connected</div>
          <div className={`conn-indicator ${wsConnected ? "ws-on" : "ws-off"}`}>
            {wsConnected ? "⚡ WebSocket Live" : "○ WebSocket Connecting..."}
          </div>
        </div>

        {lastUpdated && (
          <p className="sidebar-updated">
            Stats refreshed: {lastUpdated.toLocaleTimeString()}
          </p>
        )}

        <div className="sidebar-footer">
          <p>WaterTwinML v2.0</p>
          <p>Stack: React + DynamoDB + WebSocket</p>
        </div>
      </aside>

      <main className="main">
        <header className="header">
          <h2>WaterTwinML — Operational Dashboard</h2>
          <p className="header-sub">
            Digital Twin for Water Distribution Networks · Real-time monitoring · Anomaly detection · Risk propagation
          </p>

          <div className="kpi-row">
            <div className="kpi">
              <span className="kpi-val">{stats?.sensor_records?.toLocaleString() ?? "—"}</span>
              <span className="kpi-label">SENSOR RECORDS (24h)</span>
            </div>
            <div className="kpi">
              <span className="kpi-val kpi-live">{liveAnomalyCount.toLocaleString()}</span>
              <span className="kpi-label">ANOMALIES DETECTED</span>
            </div>
            <div className="kpi">
              <span className="kpi-val">{wsAnomalies.length}</span>
              <span className="kpi-label">LIVE (THIS SESSION)</span>
            </div>
            <div className="kpi">
              <span className={`kpi-val ${wsConnected ? "kpi-green" : "kpi-yellow"}`}>
                {wsConnected ? "LIVE" : "POLLING"}
              </span>
              <span className="kpi-label">DATA MODE</span>
            </div>
          </div>

          <nav className="tabs">
            {TABS.map((tab, i) => (
              <button
                key={i}
                className={`tab-btn ${activeTab === i ? "active" : ""}`}
                onClick={() => setActiveTab(i)}
              >
                {tab}
              </button>
            ))}
          </nav>
        </header>

        <section className="content">
          {activeTab === 0 && <AnomalyMonitoring tenantId={tenant} wsAnomalies={wsAnomalies} />}
          {activeTab === 1 && <NetworkRisk tenantId={tenant} intervalMs={30000} />}
          {activeTab === 2 && <MaintenanceAlerts tenantId={tenant} wsAnomalies={wsAnomalies} />}
          {activeTab === 3 && <QAMetrics />}
        </section>
      </main>
    </div>
  );
}
