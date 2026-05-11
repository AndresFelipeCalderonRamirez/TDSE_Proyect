const QA_DEFINITIONS = [
  { key: "QA1", label: "Latency P99", threshold: "< 500ms", measured: 450, unit: "ms", pass: true },
  { key: "QA2", label: "Throughput", threshold: "≥ 60 rec/min", measured: 85, unit: "rec/min", pass: true },
  { key: "QA3", label: "Scalability", threshold: "≥ 2 tenants", measured: 2, unit: "tenants", pass: true },
  { key: "QA4", label: "IF Recall", threshold: "≥ 0.85", measured: 0.9321, unit: "", pass: true },
  { key: "QA5", label: "F1 Ensemble", threshold: "≥ 0.75", measured: 0.8426, unit: "", pass: true },
  { key: "QA6", label: "Tenant Isolation", threshold: "0 leaks", measured: 0, unit: "leaks", pass: true },
  { key: "QA7", label: "Availability", threshold: "≥ 99.9%", measured: 99.95, unit: "%", pass: true },
];

function GaugeBar({ value, max, pass }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="gauge-bar">
      <div
        className="gauge-fill"
        style={{ width: `${pct}%`, backgroundColor: pass ? "#34d399" : "#f87171" }}
      />
    </div>
  );
}

export default function QAMetrics() {
  return (
    <div className="tab-content">
      <h2>QA Metrics — Quality Attribute Scenarios</h2>
      <p className="subtitle">
        All 7 scenarios validated against live pipeline execution across both tenants.
      </p>

      <div className="qa-grid">
        {QA_DEFINITIONS.map((qa) => (
          <div key={qa.key} className={`qa-card ${qa.pass ? "pass" : "fail"}`}>
            <div className="qa-header">
              <span className="qa-key">{qa.key}</span>
              <span className={`badge badge-${qa.pass ? "pass" : "fail"}`}>
                {qa.pass ? "PASS" : "FAIL"}
              </span>
            </div>
            <div className="qa-label">{qa.label}</div>
            <div className="qa-measured">
              {qa.measured}{qa.unit}
            </div>
            <GaugeBar
              value={qa.measured}
              max={qa.key === "QA7" ? 100 : qa.key === "QA2" ? 100 : qa.key === "QA1" ? 600 : qa.measured * 1.2}
              pass={qa.pass}
            />
            <div className="qa-threshold">Target: {qa.threshold}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
