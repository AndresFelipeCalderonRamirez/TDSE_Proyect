import { useState, useEffect, useRef, useCallback } from "react";

const WS_URL = import.meta.env.VITE_WS_URL || "";
const RECONNECT_DELAY_MS = 3000;
const MAX_RECORDS = 200;

export function useWebSocket(tenantId) {
  const [anomalies, setAnomalies] = useState([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const tenantRef = useRef(tenantId);
  tenantRef.current = tenantId;

  const connect = useCallback(() => {
    if (!WS_URL) return;

    const url = `${WS_URL}?tenant_id=${tenantRef.current}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "sensor_record" && msg.data?.isAnomaly) {
          setAnomalies((prev) => {
            // Prepend new record, keep last MAX_RECORDS
            const next = [msg.data, ...prev];
            return next.slice(0, MAX_RECORDS);
          });
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      setConnected(false);
      // Auto-reconnect
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // Reconnect when tenant changes
  useEffect(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
  }, [tenantId]);

  return { anomalies, connected };
}
