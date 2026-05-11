"""
ml/leakdb_adapter.py

Converts KIOS-Research/LeakDB data to the 8-feature JSON format consumed
by WaterTwinML (paper Section 5.3).

Feature vector produced
───────────────────────
  [pressure, flow, vibration*, delta_pressure†, diameter, material_enc, length, age]
  * vibration = rolling-std of pressure (proxy — EPANET has no vibration sensor)
  † delta_pressure computed inside WaterTwinML._build_feature_matrix

Supported LeakDB layouts
────────────────────────
Layout A — scenario folders (default):
  LeakDB/
    Scenario-1/
      Pressures.csv   rows=timestep, cols=sensor nodes
      Demands.csv     rows=timestep, cols=nodes        (optional)
      Labels.csv      rows=timestep, col "Leak" 0/1
      Leakages.csv    per-leak metadata                (optional)
    Scenario-2/  ...
    Network/
      *.inp           EPANET topology                  (optional)

Layout B — flat files (single scenario at repo root):
  LeakDB/
    Pressures.csv
    Demands.csv   (optional)
    Labels.csv

Output JSON (same format as local_simulator.py)
───────────────────────────────────────────────
  [
    { "tenant_id":  "tenant-A",
      "city":       "L-Town",
      "segment_id": "SEG_A_001",
      "timestamp":  "2019-01-01T00:00:00",
      "pressure":   45.2,
      "flow":       12.3,
      "vibration":  0.45,
      "is_anomaly": 0,
      "metadata":   { "diameter": 150.0, "material": "DI",
                      "length": 320.0, "age": 25.0, "elevation": 100.0 } },
    ...
  ]

Usage
─────
  # single scenario, all defaults
  python ml/leakdb_adapter.py \\
      --leakdb-path /path/to/LeakDB \\
      --tenant tenant-A \\
      --output data/outputs/tenant-A_sensor_data.json

  # pick scenario 3, first 72 h, with network .inp
  python ml/leakdb_adapter.py \\
      --leakdb-path /path/to/LeakDB \\
      --scenario 3 \\
      --tenant tenant-A \\
      --hours 72 \\
      --inp-file /path/to/LeakDB/Network/L-Town.inp \\
      --output data/outputs/tenant-A_sensor_data.json

  # concatenate scenarios 1-5 for a larger training set
  python ml/leakdb_adapter.py \\
      --leakdb-path /path/to/LeakDB \\
      --scenarios 1,2,3,4,5 \\
      --tenant tenant-A \\
      --output data/outputs/tenant-A_sensor_data.json
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Hazen-Williams C coefficient → (material_name, material_enc)
# Higher C = smoother pipe = newer / plastic
_C_TO_MATERIAL: List[Tuple[int, int, str, int]] = [
    # (C_min, C_max, name, enc)
    (140, 999, "PVC", 0),
    (120, 139, "DI",  2),
    (100, 119, "CI",  3),
    (0,    99, "AC",  1),
]

# Default pipe properties when .inp is unavailable
_DEFAULT_PIPE = {
    "diameter":  150.0,   # mm
    "material":  "DI",
    "material_enc": 2,
    "length":    300.0,   # m
    "age":        25.0,   # years
    "elevation":  50.0,   # m
}

# Rolling window (timesteps) for vibration proxy
_VIB_WINDOW = 12  # ≈ 1 h at 5-min resolution


# ── EPANET .inp parser ────────────────────────────────────────────────────────

def parse_inp_file(inp_path: str) -> Tuple[Dict, Dict]:
    """
    Parse EPANET .inp and return:
      pipes   : { pipe_id  -> {diameter, length, roughness, node1, node2} }
      junctions: { node_id -> {elevation} }
    Only [PIPES] and [JUNCTIONS] sections are read; everything else ignored.
    """
    pipes: Dict[str, Dict] = {}
    junctions: Dict[str, Dict] = {}
    section = None

    with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue

            if line.startswith("["):
                section = line.upper()
                continue

            if section == "[PIPES]":
                # ID  Node1  Node2  Length  Diameter  Roughness  MinorLoss  Status
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        pipes[parts[0]] = {
                            "node1":     parts[1],
                            "node2":     parts[2],
                            "length":    float(parts[3]),
                            "diameter":  float(parts[4]),
                            "roughness": float(parts[5]),
                        }
                    except ValueError:
                        pass

            elif section == "[JUNCTIONS]":
                # ID  Elevation  Demand  Pattern
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        junctions[parts[0]] = {"elevation": float(parts[1])}
                    except ValueError:
                        pass

    logger.info(
        f"Parsed .inp — {len(pipes)} pipes, {len(junctions)} junctions: {inp_path}"
    )
    return pipes, junctions


def _roughness_to_material(roughness: float) -> Tuple[str, int]:
    """Map Hazen-Williams C (roughness) to (material_name, material_enc)."""
    c = int(roughness)
    for c_min, c_max, name, enc in _C_TO_MATERIAL:
        if c_min <= c <= c_max:
            return name, enc
    return "DI", 2  # safe default


def _material_to_age(material: str) -> float:
    """Estimate typical pipe age in years from material."""
    return {"PVC": 10.0, "HDPE": 8.0, "DI": 25.0, "CI": 45.0, "AC": 50.0}.get(
        material, 20.0
    )


def build_node_metadata(
    node_ids: List[str],
    pipes: Optional[Dict] = None,
    junctions: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """
    Assign pipe metadata to each sensor node.
    Priority:
      1. Properties of the first pipe that connects to the node (from .inp)
      2. _DEFAULT_PIPE values as fallback
    """
    node_meta: Dict[str, Dict] = {}

    # Build node→pipes adjacency from .inp if available
    node_to_pipes: Dict[str, List[Dict]] = {n: [] for n in node_ids}
    if pipes:
        for p_data in pipes.values():
            for endpoint in ("node1", "node2"):
                nid = p_data[endpoint]
                if nid in node_to_pipes:
                    node_to_pipes[nid].append(p_data)

    for node_id in node_ids:
        connected = node_to_pipes.get(node_id, [])

        if connected:
            p = connected[0]
            material, material_enc = _roughness_to_material(p["roughness"])
            elevation = (
                junctions[node_id]["elevation"]
                if (junctions and node_id in junctions)
                else _DEFAULT_PIPE["elevation"]
            )
            node_meta[node_id] = {
                "diameter":     p["diameter"],
                "material":     material,
                "material_enc": material_enc,
                "length":       p["length"],
                "age":          _material_to_age(material),
                "elevation":    elevation,
            }
        else:
            # No pipe found → defaults + elevation from junctions if present
            meta = dict(_DEFAULT_PIPE)
            if junctions and node_id in junctions:
                meta["elevation"] = junctions[node_id]["elevation"]
            node_meta[node_id] = meta

    return node_meta


# ── CSV readers ───────────────────────────────────────────────────────────────

def _read_timeseries_csv(path: str) -> pd.DataFrame:
    """
    Read a LeakDB timeseries CSV (Pressures.csv / Demands.csv).
    Handles:
      - First column as timestamp string or relative time (HH:MM:SS / integer)
      - Semicolon or comma separator
    Returns DataFrame with a proper DatetimeIndex and float columns.
    """
    # Auto-detect separator
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","

    df = pd.read_csv(path, sep=sep, encoding="utf-8", errors="replace")
    df.columns = [c.strip() for c in df.columns]

    # First column → index (timestamps or step numbers)
    time_col = df.columns[0]
    df = df.set_index(time_col)

    # Convert numeric strings to float
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def _read_labels_csv(path: str) -> pd.Series:
    """
    Read Labels.csv. Expects a column named 'Leak', 'leak', 'Label', or similar.
    Returns a Series of int (0/1) indexed the same way as Pressures.csv.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","

    df = pd.read_csv(path, sep=sep, encoding="utf-8", errors="replace")
    df.columns = [c.strip() for c in df.columns]

    # Find the label column (case-insensitive)
    label_col = next(
        (c for c in df.columns if c.lower() in ("leak", "label", "anomaly", "is_leak")),
        df.columns[-1],
    )

    time_col = df.columns[0]
    df = df.set_index(time_col)
    return df[label_col].fillna(0).astype(int)


def _build_timestamps(n: int, start: str = "2019-01-01", freq_min: int = 5) -> pd.DatetimeIndex:
    """Generate synthetic timestamps when none are present in the CSV."""
    start_dt = pd.Timestamp(start)
    return pd.date_range(start=start_dt, periods=n, freq=f"{freq_min}min")


# ── Core adapter ──────────────────────────────────────────────────────────────

class LeakDBAdapter:
    """Convert a LeakDB scenario directory to WaterTwinML JSON."""

    def __init__(self, tenant_id: str, city: str = "L-Town"):
        self.tenant_id = tenant_id
        self.tenant_short = tenant_id.split("-")[1].upper()
        self.city = city

    # ── public API ────────────────────────────────────────────────────────────

    def adapt_scenario(
        self,
        scenario_dir: str,
        inp_path: Optional[str] = None,
        hours: Optional[int] = None,
        start_ts: str = "2019-01-01",
    ) -> List[Dict]:
        """
        Convert one LeakDB scenario folder to a list of records compatible
        with WaterTwinML._load_from_json().

        Parameters
        ----------
        scenario_dir : path to Scenario-N/ folder (or repo root for flat layout)
        inp_path     : optional EPANET .inp file for pipe metadata
        hours        : keep only the first N hours of data (None = all)
        start_ts     : base timestamp if CSV has no date column
        """
        scenario_dir = Path(scenario_dir)
        logger.info(f"Adapting scenario: {scenario_dir}")

        pressures_path = self._find_file(scenario_dir, "Pressures")
        demands_path   = self._find_file(scenario_dir, "Demands", required=False)
        labels_path    = self._find_file(scenario_dir, "Labels")

        if pressures_path is None or labels_path is None:
            raise FileNotFoundError(
                f"Pressures.csv or Labels.csv not found in {scenario_dir}"
            )

        # Load time-series
        pressures = _read_timeseries_csv(str(pressures_path))
        labels    = _read_labels_csv(str(labels_path))
        demands   = _read_timeseries_csv(str(demands_path)) if demands_path else None

        # Align lengths
        n = min(len(pressures), len(labels))
        pressures = pressures.iloc[:n]
        labels    = labels.iloc[:n]
        if demands is not None:
            demands = demands.iloc[:n]

        # Truncate to requested hours
        if hours is not None:
            freq_min = self._infer_freq_min(pressures)
            steps = int(hours * 60 / freq_min)
            pressures = pressures.iloc[:steps]
            labels    = labels.iloc[:steps]
            if demands is not None:
                demands = demands.iloc[:steps]
            n = len(pressures)

        # Build timestamp index
        freq_min   = self._infer_freq_min(pressures)
        timestamps = _build_timestamps(n, start=start_ts, freq_min=freq_min)

        # Pipe / node metadata from .inp
        node_ids = list(pressures.columns)
        if inp_path and os.path.exists(inp_path):
            pipes, junctions = parse_inp_file(inp_path)
        else:
            pipes, junctions = None, None
            if inp_path:
                logger.warning(f".inp not found: {inp_path} — using default metadata")

        node_meta = build_node_metadata(node_ids, pipes, junctions)

        # Vibration proxy: rolling std of pressure per node
        vib_df = pressures.rolling(_VIB_WINDOW, min_periods=1).std().fillna(0.0)

        # Build records
        records: List[Dict] = []
        label_arr = labels.values

        for step_i, ts in enumerate(timestamps):
            for node_j, node_id in enumerate(node_ids):
                seg_id   = f"SEG_{self.tenant_short}_{node_id}"
                pressure = float(pressures.iat[step_i, node_j])
                flow     = float(
                    demands.iat[step_i, node_j] if demands is not None
                    else max(0.0, pressure ** 0.5 * 0.15)  # Torricelli proxy
                )
                vibration = float(vib_df.iat[step_i, node_j])
                is_anomaly = int(label_arr[step_i])  # network-wide label

                meta = node_meta.get(node_id, dict(_DEFAULT_PIPE))

                records.append(
                    {
                        "tenant_id":  self.tenant_id,
                        "city":       self.city,
                        "segment_id": seg_id,
                        "timestamp":  ts.isoformat(),
                        "pressure":   max(0.0, pressure),
                        "flow":       max(0.0, flow),
                        "vibration":  max(0.0, vibration),
                        "is_anomaly": is_anomaly,
                        "metadata":   {
                            "diameter":  meta["diameter"],
                            "material":  meta["material"],
                            "length":    meta["length"],
                            "age":       meta["age"],
                            "elevation": meta["elevation"],
                        },
                    }
                )

        logger.info(
            f"Scenario done — {len(timestamps)} timesteps × {len(node_ids)} nodes "
            f"= {len(records):,} records | "
            f"anomaly steps: {int(label_arr[:n].sum())}"
        )
        return records

    def adapt_multiple_scenarios(
        self,
        leakdb_root: str,
        scenario_ids: List[int],
        inp_path: Optional[str] = None,
        hours_per_scenario: Optional[int] = None,
    ) -> List[Dict]:
        """Concatenate multiple scenario adaptations."""
        all_records: List[Dict] = []
        root = Path(leakdb_root)

        for sid in scenario_ids:
            # Try common folder name patterns
            for pattern in (f"Scenario-{sid}", f"Scenario_{sid}", str(sid)):
                scenario_dir = root / pattern
                if scenario_dir.exists():
                    break
            else:
                logger.warning(f"Scenario {sid} folder not found in {root} — skipping")
                continue

            try:
                records = self.adapt_scenario(
                    str(scenario_dir),
                    inp_path=inp_path,
                    hours=hours_per_scenario,
                )
                all_records.extend(records)
            except Exception as e:
                logger.warning(f"Scenario {sid} failed: {e}")

        logger.info(f"Total records after merging {len(scenario_ids)} scenarios: {len(all_records):,}")
        return all_records

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_file(directory: Path, stem: str, required: bool = True) -> Optional[Path]:
        """Find a CSV file whose name contains `stem` (case-insensitive)."""
        for p in directory.iterdir():
            if stem.lower() in p.name.lower() and p.suffix.lower() == ".csv":
                return p
        if required:
            logger.warning(f"File with stem '{stem}' not found in {directory}")
        return None

    @staticmethod
    def _infer_freq_min(df: pd.DataFrame, default: int = 5) -> int:
        """
        Guess sampling frequency in minutes from the index of the DataFrame.
        Falls back to `default` if the index is not parseable as time.
        """
        idx = df.index
        if len(idx) < 2:
            return default
        # Try parsing as HH:MM:SS or numeric
        try:
            t0 = pd.to_timedelta(str(idx[0]))
            t1 = pd.to_timedelta(str(idx[1]))
            mins = int((t1 - t0).total_seconds() / 60)
            return mins if mins > 0 else default
        except Exception:
            pass
        # Numeric index: assume default
        return default


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LeakDB data to WaterTwinML 8-feature JSON format"
    )
    parser.add_argument(
        "--leakdb-path", required=True,
        help="Root folder of the cloned LeakDB repository",
    )
    parser.add_argument(
        "--tenant", default="tenant-A",
        choices=["tenant-A", "tenant-B"],
        help="Target tenant ID (default: tenant-A)",
    )
    parser.add_argument(
        "--city", default="L-Town",
        help="City label stored in JSON records (default: L-Town)",
    )

    # Scenario selection (mutually exclusive)
    scenario_group = parser.add_mutually_exclusive_group()
    scenario_group.add_argument(
        "--scenario", type=int, default=1,
        help="Single scenario number to convert (default: 1)",
    )
    scenario_group.add_argument(
        "--scenarios", type=str,
        help="Comma-separated list of scenario numbers, e.g. '1,2,3'",
    )

    parser.add_argument(
        "--hours", type=int, default=None,
        help="Keep only the first N hours of each scenario (default: all)",
    )
    parser.add_argument(
        "--inp-file", type=str, default=None,
        help="Path to EPANET .inp file for pipe metadata (optional)",
    )
    parser.add_argument(
        "--output", type=str,
        default=None,
        help="Output JSON path (default: data/outputs/<tenant>_sensor_data.json)",
    )
    parser.add_argument(
        "--start-ts", type=str, default="2019-01-01",
        help="Start timestamp for generated index (default: 2019-01-01)",
    )

    args = parser.parse_args()

    # Resolve output path
    if args.output is None:
        os.makedirs("data/outputs", exist_ok=True)
        args.output = f"data/outputs/{args.tenant}_sensor_data.json"

    adapter = LeakDBAdapter(tenant_id=args.tenant, city=args.city)
    root = Path(args.leakdb_path)

    # ── Single scenario
    if args.scenarios is None:
        # Try scenario subfolder first, then root (Layout B)
        for pattern in (
            f"Scenario-{args.scenario}",
            f"Scenario_{args.scenario}",
            str(args.scenario),
        ):
            scenario_dir = root / pattern
            if scenario_dir.exists():
                break
        else:
            scenario_dir = root  # flat layout

        records = adapter.adapt_scenario(
            str(scenario_dir),
            inp_path=args.inp_file,
            hours=args.hours,
            start_ts=args.start_ts,
        )

    # ── Multiple scenarios
    else:
        ids = [int(x.strip()) for x in args.scenarios.split(",")]
        records = adapter.adapt_multiple_scenarios(
            str(root),
            scenario_ids=ids,
            inp_path=args.inp_file,
            hours_per_scenario=args.hours,
        )

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote {len(records):,} records to {args.output}")
    logger.info("Next step:")
    logger.info(
        f"  python ml/train_models.py --train-ensemble --tenant {args.tenant} "
        f"--data-path {args.output}"
    )


if __name__ == "__main__":
    main()
