"""
Ingestion of spatial wafer scans (contour/radial GR/Ge% measurements, e.g.
from SE contour mapping or XRD radial line scans) into the WaferScan sibling
data model (see spatial.py module docstring for why this is a PARALLEL path
to schema.py's CanonicalRow/Dataset, not a bolt-on field).

TWO-FILE LONG FORMAT:
  runs_csv:   one row per physical wafer run --
              run_id, T_set_C, HCl_over_DCS, GeH4_over_DCS,
              B2H6_over_DCS (optional), growth_time_s (optional), plus any
              number of Stick_<n> (nozzle flow, sccm) and probe_<n>
              (temperature, C) columns -- captured generically (any column
              starting with "Stick_"/"probe_"), since the number of
              nozzles/probes varies by reactor.
  points_csv: one row per measured location --
              run_id, x_mm, y_mm (pass y_mm=0 for a pure radial line scan,
              e.g. an XRD scan from (0,0) to (145,0)), GR_nm_min_local
              (optional), Ge_at_pct_local (optional, 0-100),
              thickness_A_local (optional), measurement_source (optional
              free-text tag, e.g. "SE_contour" or "XRD_radial").
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from chem_ml.schema import ChemClass, _c_to_k
from chem_ml.spatial import WaferPoint, WaferRunMeta, WaferScan

_NOZZLE_PREFIX = "Stick_"
_PROBE_PREFIX = "probe_"


def ingest_wafer_scan_csv(runs_csv: str | Path, points_csv: str | Path,
                         reactor_id: str, chem_class: ChemClass,
                         source_tag: str = "") -> list[WaferScan]:
    """Parse one runs.csv + one points.csv into a list of WaferScan (one per
    run_id present in runs_csv). reactor_id/chem_class are caller-declared,
    never inferred from the CSV -- same explicit-tagging convention as
    ingest_standard_csv (schema.py), for the same anti-contamination reason
    (see data_store.py module docstring)."""
    runs_df = pd.read_csv(runs_csv)
    points_df = pd.read_csv(points_csv)

    scans: list[WaferScan] = []
    for r in runs_df.itertuples():
        nozzle_flows = {c: float(getattr(r, c)) for c in runs_df.columns if c.startswith(_NOZZLE_PREFIX)}
        probe_temps = {c: _c_to_k(float(getattr(r, c))) for c in runs_df.columns if c.startswith(_PROBE_PREFIX)}
        meta = WaferRunMeta(
            run_id=str(r.run_id), reactor_id=reactor_id, chem_class=chem_class,
            T_set_K=_c_to_k(r.T_set_C), p_DCS=1.0, p_GeH4=r.GeH4_over_DCS, p_HCl=r.HCl_over_DCS,
            p_B2H6=getattr(r, "B2H6_over_DCS", 0.0) or 0.0,
            nozzle_flows_sccm=nozzle_flows, probe_temps_K=probe_temps,
            growth_time_s=getattr(r, "growth_time_s", None),
            source_dataset=source_tag or f"wafer_scan:{Path(runs_csv).stem}",
        )

        pts_df = points_df[points_df["run_id"].astype(str) == str(r.run_id)]
        points = []
        for p in pts_df.itertuples():
            ge_pct = getattr(p, "Ge_at_pct_local", None)
            points.append(WaferPoint(
                run_id=str(r.run_id), x_mm=float(p.x_mm), y_mm=float(getattr(p, "y_mm", 0.0) or 0.0),
                GR_nm_min_local=getattr(p, "GR_nm_min_local", None),
                Ge_at_frac_local=(ge_pct / 100.0 if ge_pct is not None else None),
                thickness_A_local=getattr(p, "thickness_A_local", None),
                measurement_source=getattr(p, "measurement_source", "") or "",
            ))

        scan = WaferScan(meta=meta, points=points)
        scan.validate()
        scans.append(scan)
    return scans
