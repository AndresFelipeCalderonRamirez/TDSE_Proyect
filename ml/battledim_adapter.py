"""
ml/battledim_adapter.py

Converts BattLeDIM Dataset Generator output (Measurements.xlsx) to the
8-feature JSON format consumed by WaterTwinML (paper Section 5.3).

Workflow
────────
Step 1 — generate the data (run once, ~2 min for 72 h config):
    cd "C:\\Users\\Geme\\Desktop\\BattLeDIM\\Dataset Generator"
    pip install wntr pyyaml xlsxwriter openpyxl
    copy dataset_configuration_72h.yalm dataset_configuration.yalm
    python dataset_generator.py
    # Creates: Results\\Measurements.xlsx

Step 2 — adapt to 8-feature JSON and train:
    cd C:\\Users\\Geme\\Desktop\\TDSE_Proyect
    python ml/battledim_adapter.py ^
        --measurements "C:\\Users\\Geme\\Desktop\\BattLeDIM\\Dataset Generator\\Results\\Measurements.xlsx" ^
        --config "C:\\Users\\Geme\\Desktop\\BattLeDIM\\Dataset Generator\\dataset_configuration_72h.yalm" ^
        --inp network/L-TOWN_v2_Real.inp ^
        --tenant tenant-A ^
        --output data/outputs/tenant-A_sensor_data.json

    python ml/train_models.py --train-ensemble --tenant tenant-A

Feature mapping
───────────────
  pressure     <- Pressures (m) sheet, per sensor node
  flow         <- Demands (L_h) sheet per node; PUMP_1 flow if node not in AMR
  vibration    <- rolling-std of pressure (window=12 steps, proxy for pipe stress)
  delta_pressure <- computed by WaterTwinML._build_feature_matrix
  diameter     <- [PIPES] section of .inp (first pipe adjacent to node)
  material_enc <- Hazen-Williams roughness -> PVC=0, AC=1, DI=2, CI=3
  length       <- [PIPES] section of .inp
  age          <- estimated from material type
  is_anomaly   <- 1 during any active leak period (from .yalm config), else 0
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Material encoding (paper Section 5.3) ────────────────────────────────────
# Hazen-Williams C -> (name, enc)
_C_RANGES: List[Tuple[int, int, str, int]] = [
    (140, 999, "PVC", 0),
    (120, 139, "DI",  2),
    (100, 119, "CI",  3),
    (0,    99, "AC",  1),
]

_MATERIAL_AGE = {"PVC": 10.0, "DI": 25.0, "CI": 45.0, "AC": 50.0}

_DEFAULT_PIPE = {
    "diameter": 150.0, "material": "DI", "material_enc": 2,
    "length": 300.0, "age": 25.0, "elevation": 50.0,
}

_VIB_WINDOW = 12   # rolling-std window (12 × 5 min = 1 h)


# ── .inp parser ───────────────────────────────────────────────────────────────

def parse_inp(inp_path: str) -> Tuple[Dict, Dict]:
    """Return (pipes, junctions) dicts from an EPANET .inp file."""
    pipes: Dict[str, Dict] = {}
    junctions: Dict[str, Dict] = {}
    section = None

    with open(inp_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("["):
                section = line.upper()
                continue
            parts = line.split()
            if section == "[PIPES]" and len(parts) >= 6:
                try:
                    pipes[parts[0]] = {
                        "node1": parts[1], "node2": parts[2],
                        "length": float(parts[3]), "diameter": float(parts[4]),
                        "roughness": float(parts[5]),
                    }
                except ValueError:
                    pass
            elif section == "[JUNCTIONS]" and len(parts) >= 2:
                try:
                    junctions[parts[0]] = {"elevation": float(parts[1])}
                except ValueError:
                    pass

    logger.info(f"Parsed .inp: {len(pipes)} pipes, {len(junctions)} junctions")
    return pipes, junctions


def _roughness_to_material(c: float) -> Tuple[str, int]:
    for lo, hi, name, enc in _C_RANGES:
        if lo <= int(c) <= hi:
            return name, enc
    return "DI", 2


def build_node_metadata(
    node_ids: List[str],
    pipes: Optional[Dict],
    junctions: Optional[Dict],
) -> Dict[str, Dict]:
    """Assign pipe metadata to each pressure-sensor node from .inp data."""
    node_to_pipes: Dict[str, List[Dict]] = {n: [] for n in node_ids}
    if pipes:
        for pd_item in pipes.values():
            for ep in ("node1", "node2"):
                nid = pd_item[ep]
                if nid in node_to_pipes:
                    node_to_pipes[nid].append(pd_item)

    meta: Dict[str, Dict] = {}
    for nid in node_ids:
        connected = node_to_pipes.get(nid, [])
        if connected:
            p = connected[0]
            mat, enc = _roughness_to_material(p["roughness"])
            elev = (
                junctions[nid]["elevation"]
                if junctions and nid in junctions
                else _DEFAULT_PIPE["elevation"]
            )
            meta[nid] = {
                "diameter": p["diameter"], "material": mat, "material_enc": enc,
                "length": p["length"], "age": _MATERIAL_AGE.get(mat, 20.0),
                "elevation": elev,
            }
        else:
            m = dict(_DEFAULT_PIPE)
            if junctions and nid in junctions:
                m["elevation"] = junctions[nid]["elevation"]
            meta[nid] = m

    return meta


# ── .yalm label builder ───────────────────────────────────────────────────────

def build_anomaly_labels(config_path: str, timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Returns a binary array (len = len(timestamps)).
    is_anomaly[i] = 1 if ANY leak is active at timestamps[i], else 0.
    Leak periods are read from the leakages section of the .yalm config.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    leakages = cfg.get("leakages", [])
    # First item is a comment dict — skip non-string entries
    leak_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    for entry in leakages:
        if not isinstance(entry, str):
            continue
        parts = [p.strip() for p in entry.split(",")]
        if len(parts) < 3:
            continue
        try:
            t_start = pd.Timestamp(parts[1])
            t_end   = pd.Timestamp(parts[2])
            leak_intervals.append((t_start, t_end))
        except Exception:
            pass

    labels = np.zeros(len(timestamps), dtype=np.int8)
    for t_start, t_end in leak_intervals:
        mask = (timestamps >= t_start) & (timestamps <= t_end)
        labels[mask] = 1

    n_leak = int(labels.sum())
    logger.info(
        f"Leak labels: {n_leak} anomaly steps / {len(timestamps)} total "
        f"({n_leak/len(timestamps)*100:.1f}%)"
    )
    return labels


# ── Excel reader ──────────────────────────────────────────────────────────────

def read_measurements(xlsx_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read BattLeDIM Measurements.xlsx.
    Returns (pressures_df, demands_df) with DatetimeIndex.
    Columns = sensor / node IDs.
    """
    logger.info(f"Reading {xlsx_path} ...")
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")

    def _load_sheet(name_pattern: str) -> pd.DataFrame:
        sheet = next(
            (s for s in xl.sheet_names if name_pattern.lower() in s.lower()), None
        )
        if sheet is None:
            return pd.DataFrame()
        df = xl.parse(sheet)
        # First column is Timestamp
        ts_col = df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
        df.index.name = "timestamp"
        return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    pressures = _load_sheet("Pressure")
    demands   = _load_sheet("Demand")

    logger.info(
        f"Pressures: {pressures.shape} | Demands: {demands.shape}"
    )
    return pressures, demands


# ── Core adapter ──────────────────────────────────────────────────────────────

class BattleDIMAdapter:
    def __init__(self, tenant_id: str, city: str = "L-Town"):
        self.tenant_id  = tenant_id
        self.tenant_short = tenant_id.split("-")[1].upper()
        self.city = city

    def adapt(
        self,
        measurements_path: str,
        config_path: str,
        inp_path: Optional[str] = None,
    ) -> List[Dict]:
        """
        Convert Measurements.xlsx + .yalm config to WaterTwinML JSON records.
        Each (timestep × pressure-sensor node) becomes one record.
        """
        pressures, demands = read_measurements(measurements_path)

        if pressures.empty:
            raise ValueError(
                "Pressures sheet not found in Measurements.xlsx. "
                "Run dataset_generator.py first."
            )

        timestamps = pressures.index
        node_ids   = list(pressures.columns)

        # Leak labels from config
        labels = build_anomaly_labels(config_path, timestamps)

        # Pipe metadata from .inp
        if inp_path and os.path.exists(inp_path):
            pipes, junctions = parse_inp(inp_path)
        else:
            pipes, junctions = None, None
            if inp_path:
                logger.warning(f".inp not found: {inp_path} — using default metadata")

        node_meta = build_node_metadata(node_ids, pipes, junctions)

        # Vibration proxy: rolling std of pressure (hydraulic noise ≈ pipe stress)
        vib_df = pressures.rolling(_VIB_WINDOW, min_periods=1).std().fillna(0.0)

        # Build records
        records: List[Dict] = []
        for step_i, ts in enumerate(timestamps):
            for node_j, node_id in enumerate(node_ids):
                pressure  = float(pressures.iat[step_i, node_j])
                vibration = float(vib_df.iat[step_i, node_j])

                # Flow: use demand for this node if available, else 0
                if not demands.empty and node_id in demands.columns:
                    flow = float(demands.at[ts, node_id])
                else:
                    # Torricelli proxy: flow ∝ sqrt(pressure head)
                    flow = float(max(0.0, pressure ** 0.5 * 0.15))

                meta = node_meta.get(node_id, dict(_DEFAULT_PIPE))

                records.append({
                    "tenant_id":  self.tenant_id,
                    "city":       self.city,
                    "segment_id": f"SEG_{self.tenant_short}_{node_id}",
                    "timestamp":  ts.isoformat(),
                    "pressure":   max(0.0, pressure),
                    "flow":       max(0.0, flow),
                    "vibration":  max(0.0, vibration),
                    "is_anomaly": int(labels[step_i]),
                    "metadata": {
                        "diameter":  meta["diameter"],
                        "material":  meta["material"],
                        "length":    meta["length"],
                        "age":       meta["age"],
                        "elevation": meta["elevation"],
                    },
                })

        n_steps = len(timestamps)
        n_nodes = len(node_ids)
        n_anom  = int(labels.sum())
        logger.info(
            f"Done: {n_steps} timesteps x {n_nodes} nodes = {len(records):,} records "
            f"| anomaly steps: {n_anom} ({n_anom/n_steps*100:.1f}%)"
        )
        return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert BattLeDIM Measurements.xlsx to WaterTwinML 8-feature JSON. "
            "Run dataset_generator.py first to create Measurements.xlsx."
        )
    )
    parser.add_argument(
        "--measurements", required=True,
        help="Path to Results/Measurements.xlsx produced by dataset_generator.py",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to dataset_configuration*.yalm (used to extract leak periods)",
    )
    parser.add_argument(
        "--inp", default="network/L-TOWN_v2_Real.inp",
        help="Path to L-TOWN_v2_Real.inp for pipe metadata (default: network/L-TOWN_v2_Real.inp)",
    )
    parser.add_argument(
        "--tenant", default="tenant-A",
        choices=["tenant-A", "tenant-B"],
    )
    parser.add_argument(
        "--city", default="L-Town",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: data/outputs/<tenant>_sensor_data.json)",
    )
    args = parser.parse_args()

    if args.output is None:
        os.makedirs("data/outputs", exist_ok=True)
        args.output = f"data/outputs/{args.tenant}_sensor_data.json"

    adapter = BattleDIMAdapter(tenant_id=args.tenant, city=args.city)
    records = adapter.adapt(
        measurements_path=args.measurements,
        config_path=args.config,
        inp_path=args.inp,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote {len(records):,} records -> {args.output}")
    logger.info("")
    logger.info("Next: train the ensemble")
    logger.info(
        f"  python ml/train_models.py --train-ensemble --tenant {args.tenant} "
        f"--data-path {args.output}"
    )


if __name__ == "__main__":
    main()
