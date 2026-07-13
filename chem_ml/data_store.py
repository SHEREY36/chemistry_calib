"""
Additive data accumulation: the model is not retrained from scratch as the
epitaxy team collects more data -- it grows the SAME canonical dataset the
Tomasini reproduction started from, and re-calibrates against the growing
whole (see cli.py `add-data` / `calibrate --pooled`).

WHERE CONTAMINATION COULD ENTER, AND WHAT STOPS IT AT EACH POINT:

1. Double-counting a source. Re-registering the same CSV twice would
   silently double-weight those rows in the likelihood (equivalent to
   telling NUTS you're twice as sure about those conditions as you are).
   `register_new_data` refuses to re-register a `source_tag` already in
   the manifest.

2. Cross-reactor leakage into the chemistry fit. New data from a
   DIFFERENT reactor than the one theta_chem is fit on must NOT be pooled
   into that fit (see pipeline.py's run_phase4_calibration, which now
   filters by (chem_class, reactor_id) rather than a literal Tomasini
   source-tag -- see the docstring there). It should go through Phase 7's
   transfer route instead. This module does not decide that for you: it
   just accumulates rows with their DECLARED reactor_id, and it is
   run_phase4_calibration's filter, plus the caller choosing calibrate
   (chemistry fit) vs. add-reactor (transfer fit), that keeps reactor
   effects from leaking into theta_chem. Get the reactor_id tag right at
   registration time -- everything downstream trusts it.

3. Cross-class leakage into GR/Ge/B. New data of a different chem_class
   (e.g. SiGe:P, phosphine-doped) must not perturb the existing SiGe/
   SiGe:B fits. This is enforced structurally by
   ReactionNetworkAssembler's hard class gate (Phase 2.3) AND by
   run_phase4_calibration's chem_class filter -- adding SiGe:P data changes
   nothing about the SiGe/SiGe:B pipelines until a SiGe:P-specific
   sub-model is added (a new, structurally separate slot -- see
   registry.py/assembler.py for the pattern to follow). Verified directly
   in tests/test_data_store.py's anti-contamination test.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from chem_ml.config import Config
from chem_ml.schema import ChemClass, Dataset, Mode, ingest_standard_csv, ingest_tomasini

log = logging.getLogger("chem_ml")


@dataclass(frozen=True)
class RegisteredSource:
    source_tag: str
    csv_path: str
    reactor_id: str
    chem_class: str
    mode: str
    added_at: str


def _manifest_path(cfg: Config) -> Path:
    return Path(cfg.data_processed) / "additions_manifest.json"


def _load_manifest(cfg: Config) -> list[dict]:
    p = _manifest_path(cfg)
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _save_manifest(cfg: Config, manifest: list[dict]) -> None:
    p = _manifest_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))


def registered_source_tags(cfg: Config) -> set[str]:
    return {e["source_tag"] for e in _load_manifest(cfg)}


def register_new_data(cfg: Config, csv_path: str, reactor_id: str, chem_class: ChemClass,
                      source_tag: str, mode: Mode = Mode.BLANKET) -> None:
    """Validate and record a new data source in the manifest. Raises if
    `source_tag` was already registered (see module docstring, point 1) --
    this is a deliberate hard failure, not a silent skip, so a re-run
    can't accidentally slip past it."""
    existing = registered_source_tags(cfg)
    if source_tag in existing:
        raise ValueError(
            f"source_tag '{source_tag}' is already registered (see "
            f"{_manifest_path(cfg)}). Use a new, unique tag per data drop -- "
            f"re-registering the same tag would double-count those rows."
        )
    # Validate it parses and passes schema checks BEFORE writing to the
    # manifest -- a bad file should never get memorialized as "added".
    ingest_standard_csv(csv_path, reactor_id=reactor_id, chem_class=chem_class,
                       mode=mode, source_tag=source_tag)

    from datetime import datetime, timezone
    manifest = _load_manifest(cfg)
    manifest.append(asdict(RegisteredSource(
        source_tag=source_tag, csv_path=str(Path(csv_path).resolve()),
        reactor_id=reactor_id, chem_class=chem_class.value, mode=mode.value,
        added_at=datetime.now(timezone.utc).isoformat(),
    )))
    _save_manifest(cfg, manifest)
    log.info("Registered new data source '%s' (%s, %s, reactor=%s)",
             source_tag, csv_path, chem_class.value, reactor_id)


def load_accumulated_dataset(cfg: Config) -> Dataset:
    """Tomasini's base ingest UNION every registered addition. This is what
    `calibrate --pooled` fits against; Phase 0-8's exact reproduction
    (load_all_datasets in pipeline.py) is untouched and still returns pure
    Tomasini only, so nothing about the original STOP GATE result changes
    underfoot as data accumulates."""
    ds = ingest_tomasini(cfg.data_raw)
    for entry in _load_manifest(cfg):
        add_ds = ingest_standard_csv(
            entry["csv_path"], reactor_id=entry["reactor_id"],
            chem_class=ChemClass(entry["chem_class"]), mode=Mode(entry["mode"]),
            source_tag=entry["source_tag"],
        )
        ds = ds + add_ds
    return ds


def load_production_dataset(cfg: Config) -> Dataset:
    """Load only registered experimental data for production calibration.

    Tomasini remains available through ``ingest_tomasini`` and the benchmark
    validation path, but it is deliberately excluded here so production
    chemistry packages are trained from Applied/epitaxy-team data only.
    """
    ds = Dataset([])
    for entry in _load_manifest(cfg):
        add_ds = ingest_standard_csv(
            entry["csv_path"], reactor_id=entry["reactor_id"],
            chem_class=ChemClass(entry["chem_class"]), mode=Mode(entry["mode"]),
            source_tag=entry["source_tag"],
        )
        ds = ds + add_ds
    return ds


# ---------------------------------------------------------------------------
# Phase 12: spatial wafer-scan registration. A DELIBERATELY SEPARATE manifest
# (spatial_manifest.json, not additions_manifest.json) with the SAME
# dedup-by-source_tag guarantee as register_new_data above, so a spatial-scan
# registration can never be confused with, or silently pooled into, the
# scalar add-data/calibrate path that feeds run_phase4_calibration directly
# -- see spatial.py's module docstring for why these are deliberately
# parallel, not unified, paths.
# ---------------------------------------------------------------------------
def _spatial_manifest_path(cfg: Config) -> Path:
    return Path(cfg.data_processed) / "spatial_manifest.json"


def registered_spatial_tags(cfg: Config) -> set[str]:
    p = _spatial_manifest_path(cfg)
    if not p.exists():
        return set()
    return {e["source_tag"] for e in json.loads(p.read_text())}


def register_wafer_scan(cfg: Config, runs_csv: str, points_csv: str, reactor_id: str,
                        chem_class: ChemClass, source_tag: str) -> None:
    """Spatial-data analog of register_new_data: validate-before-writing,
    hard-refuse a re-used source_tag."""
    from chem_ml.spatial_ingest import ingest_wafer_scan_csv

    existing = registered_spatial_tags(cfg)
    if source_tag in existing:
        raise ValueError(
            f"source_tag '{source_tag}' is already registered (see "
            f"{_spatial_manifest_path(cfg)}). Use a new, unique tag per wafer scan."
        )
    # Validate it parses and passes WaferScan checks BEFORE writing to the
    # manifest -- same "never memorialize a bad file as added" discipline.
    ingest_wafer_scan_csv(runs_csv, points_csv, reactor_id=reactor_id, chem_class=chem_class,
                         source_tag=source_tag)

    from datetime import datetime, timezone
    p = _spatial_manifest_path(cfg)
    manifest = json.loads(p.read_text()) if p.exists() else []
    manifest.append({
        "source_tag": source_tag, "runs_csv": str(Path(runs_csv).resolve()),
        "points_csv": str(Path(points_csv).resolve()), "reactor_id": reactor_id,
        "chem_class": chem_class.value, "added_at": datetime.now(timezone.utc).isoformat(),
    })
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))
    log.info("Registered wafer scan '%s' (%s, %s, reactor=%s)",
             source_tag, runs_csv, chem_class.value, reactor_id)


def load_registered_wafer_scans(cfg: Config, source_tag: str) -> list:
    """Load (parse fresh from disk, same no-caching convention as
    load_accumulated_dataset) the WaferScans registered under `source_tag`."""
    from chem_ml.spatial_ingest import ingest_wafer_scan_csv

    p = _spatial_manifest_path(cfg)
    manifest = json.loads(p.read_text()) if p.exists() else []
    entry = next((e for e in manifest if e["source_tag"] == source_tag), None)
    if entry is None:
        raise ValueError(f"No wafer scan registered under source_tag={source_tag!r} (see {p})")
    return ingest_wafer_scan_csv(entry["runs_csv"], entry["points_csv"], reactor_id=entry["reactor_id"],
                                chem_class=ChemClass(entry["chem_class"]), source_tag=source_tag)
