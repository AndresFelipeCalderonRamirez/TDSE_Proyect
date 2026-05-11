import axios from "axios";

const BASE = import.meta.env.VITE_API_BASE_URL || "";

const api = axios.create({ baseURL: BASE, timeout: 10000 });

const get = (path, params) => api.get(path, { params }).then((r) => r.data);

export const fetchStats = (tenantId) =>
  get("/stats", { tenant_id: tenantId });

export const fetchAnomalies = (tenantId, limit = 100) =>
  get("/anomalies", { tenant_id: tenantId, limit });

export const fetchSensorRecords = (tenantId, limit = 100) =>
  get("/sensor-records", { tenant_id: tenantId, limit });

export const fetchRanking = (tenantId) =>
  get("/ranking", { tenant_id: tenantId });

export const fetchMaintenanceAlerts = (tenantId, limit = 100) =>
  get("/maintenance-alerts", { tenant_id: tenantId, limit });

export const fetchTopology = (tenantId) =>
  get("/topology", { tenant_id: tenantId });
