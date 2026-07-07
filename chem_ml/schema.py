"""
Canonical schema for one physical growth condition, plus ingestion of the
Tomasini et al. (2010) appendix datasets (DS1-DS4) into that schema.

NORMALIZATION CONVENTION: Tomasini's own data (and this pipeline) works in
partial pressures NORMALIZED to the Si-source precursor (Eq. 6 of the paper):
  p_DCS := 1.0 (dimensionless reference)
  p_HCl, p_GeH4, p_B2H6 := ratio to p_DCS (dimensionless), exactly as tabulated
  in the appendices. No absolute Torr reconstruction is needed or attempted --
  the physics core only ever consumes ln(p_i / p_DCS).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd


class Mode(str, Enum):
    BLANKET = "blanket"
    SELECTIVE = "selective"


class ChemClass(str, Enum):
    SI = "Si"
    SI_X = "Si:X"
    SIGE = "SiGe"
    SIGE_X = "SiGe:X"
    SIGE_B = "SiGe:B"
    SIGE_P = "SiGe:P"
    SIC = "SiC"
    SIGEC = "SiGeC"
    SIGEC_X = "SiGeC:X"


_CHEM_CLASS_ALIASES = {
    "Si": ChemClass.SI,
    "Si:X": ChemClass.SI_X,
    "SiGe": ChemClass.SIGE,
    "SiGe:X": ChemClass.SIGE_X,
    "SiGe:B": ChemClass.SIGE_B,
    "SiGe:P": ChemClass.SIGE_P,
    "SiC": ChemClass.SIC,
    "SiGeC": ChemClass.SIGEC,
    "SiGeC:X": ChemClass.SIGEC_X,
}


def parse_chem_class(value: str | ChemClass) -> ChemClass:
    """Parse public chemistry-class labels, preserving legacy aliases."""
    if isinstance(value, ChemClass):
        return value
    try:
        return _CHEM_CLASS_ALIASES[value]
    except KeyError as exc:
        raise ValueError(f"unknown chem_class {value!r}") from exc


def canonical_chem_class(value: str | ChemClass) -> ChemClass:
    """Return the model-routing class for a public chemistry-class label."""
    c = parse_chem_class(value)
    if c in (ChemClass.SIGE_B, ChemClass.SIGE_P):
        return ChemClass.SIGE_X
    return c


@dataclass
class CanonicalRow:
    """One physical condition (one wafer / one calibration point).
    Partial pressures are DIMENSIONLESS, normalized to p_DCS=1 (see module
    docstring). T in KELVIN (convert from C at ingest!)."""
    reactor_id: str
    chem_class: ChemClass
    mode: Mode
    T_K: float
    p_DCS: float
    p_GeH4: float
    p_HCl: float
    p_B2H6: float = 0.0
    p_MMS: float = 0.0
    p_dopant: float = 0.0
    p_H2: float = 0.0
    p_N2: float = 0.0
    p_carrier: float = 0.0
    XT_flow_H2_minus_N2_sccm: float = 0.0
    pattern_density: float = 0.0          # 0 for blanket
    run_id: str = ""
    si_source: str = "DCS"
    dopant_species: str = ""
    growth_time_s: Optional[float] = None
    GR_nm_min: Optional[float] = None
    Ge_at_frac: Optional[float] = None    # 0..1
    C_at_frac: Optional[float] = None     # 0..1
    B_conc: Optional[float] = None        # at/cm^3
    dopant_conc: Optional[float] = None   # at/cm^3
    dopant_at_frac: Optional[float] = None
    source_dataset: str = ""

    def validate(self) -> None:
        """Fail-loud validation (Phase 1.3). Raise on physical impossibility."""
        assert self.T_K > 273.0, f"T_K={self.T_K} looks like Celsius; convert at ingest"
        for name in ("p_DCS", "p_GeH4", "p_HCl", "p_B2H6", "p_MMS", "p_dopant", "p_H2", "p_N2"):
            v = getattr(self, name)
            assert v >= 0.0, f"{name} negative ({v})"
        assert self.p_DCS > 0.0, "p_DCS must be > 0 (normalization base)"
        if self.GR_nm_min is not None:
            assert self.GR_nm_min > 0.0, "GR must be > 0"
        if self.Ge_at_frac is not None:
            assert 0.0 < self.Ge_at_frac < 1.0, "Ge fraction out of (0,1)"
        if self.C_at_frac is not None:
            assert 0.0 < self.C_at_frac < 1.0, "C fraction out of (0,1)"
        if self.dopant_at_frac is not None:
            assert 0.0 < self.dopant_at_frac < 1.0, "dopant fraction out of (0,1)"


@dataclass
class Dataset:
    rows: list[CanonicalRow]

    def validate(self) -> None:
        for r in self.rows:
            r.validate()

    def filter(self, **kw) -> "Dataset":
        def ok(r: CanonicalRow) -> bool:
            return all(getattr(r, k) == v for k, v in kw.items())
        return Dataset([r for r in self.rows if ok(r)])

    def filter_where(self, predicate) -> "Dataset":
        """Like `filter`, but for conditions equality kwargs can't express
        (e.g. `lambda r: r.B_conc is not None`)."""
        return Dataset([r for r in self.rows if predicate(r)])

    def __add__(self, other: "Dataset") -> "Dataset":
        return Dataset(self.rows + other.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.rows])


def _c_to_k(t_c: float) -> float:
    return t_c + 273.15


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _get(row, name: str, default=None):
    v = getattr(row, name, default)
    return default if _is_missing(v) else v


def _ratio_or_flow(row, ratio_names: tuple[str, ...], flow_name: str, si_flow) -> float:
    for name in ratio_names:
        v = _get(row, name, None)
        if v is not None:
            return float(v)
    flow = _get(row, flow_name, None)
    if flow is None:
        return 0.0
    if si_flow is None or float(si_flow) <= 0.0:
        raise ValueError(f"{flow_name} was provided but Si_source_flow_sccm is missing/non-positive")
    return float(flow) / float(si_flow)


def _derive_gr_nm_min(row):
    gr = _get(row, "GR_nm_min", None)
    if gr is not None:
        return float(gr)
    thickness_nm = _get(row, "thickness_nm", None)
    growth_time_s = _get(row, "growth_time_s", None)
    if thickness_nm is not None and growth_time_s:
        return float(thickness_nm) * 60.0 / float(growth_time_s)
    thickness_A = _get(row, "thickness_A", None)
    if thickness_A is not None and growth_time_s:
        return float(thickness_A) * 6.0 / float(growth_time_s)
    return None


def _default_dopant_species(chem_class: ChemClass, row) -> str:
    explicit = _get(row, "dopant_species", "")
    if explicit:
        return str(explicit)
    if chem_class == ChemClass.SIGE_B:
        return "B2H6"
    if chem_class == ChemClass.SIGE_P:
        return "PH3"
    return ""


def ingest_tomasini(data_raw: str | Path) -> Dataset:
    """Parse the 5 transcribed Tomasini appendix CSVs into canonical rows.

    - DS1  (Appendix I, i-SiGe, 70 rows): reactor ASM_Epsilon, class SiGe.
    - DS2  (Appendix I cont., SiGe:B, 760 C): two DISTINCT row sets sharing a
      feature space -- 18 GR/Ge rows and 11 unlinked [B] rows -- both tagged
      reactor ASM_Epsilon, class SiGe:B.
    - DS3  (Appendix II, Hartmann, 35 rows, isothermal 750 C): reactor
      Hartmann, class SiGe. Ratio columns are pre-scaled by x10000 in the
      paper; divided out here.
    - DS4  (Appendix III, Tan, 18 rows): reactor Tan, class SiGe:B (B2H6
      flowed, if at trace level). Absolute sccm flows converted to
      DCS-normalized ratios (p_i/p_DCS = flow_i/flow_DCS). NO GROWTH TIME is
      given in the appendix, so GR_nm_min is left as None for all DS4 rows --
      only Thickness/Ge% are usable (Ge/Si-ratio cross-reactor check only,
      see Phase 7 notes in build_steps_and_cfd_integration.md).
    """
    root = Path(data_raw)
    rows: list[CanonicalRow] = []

    # ---- DS1: i-SiGe -------------------------------------------------------
    df1 = pd.read_csv(root / "tomasini_ds1.csv")
    for r in df1.itertuples():
        rows.append(CanonicalRow(
            reactor_id="ASM_Epsilon", chem_class=ChemClass.SIGE, mode=Mode.BLANKET,
            T_K=_c_to_k(r.Tg_C), p_DCS=1.0, p_GeH4=r.GeH4_DCS, p_HCl=r.HCl_DCS,
            GR_nm_min=r.GR_nm_min, Ge_at_frac=r.Ge_at_pct / 100.0,
            source_dataset="DS1",
        ))

    # ---- DS2: SiGe:B, GR/Ge rows (760 C, isothermal) ------------------------
    df2gr = pd.read_csv(root / "tomasini_ds2_gr.csv")
    for r in df2gr.itertuples():
        rows.append(CanonicalRow(
            reactor_id="ASM_Epsilon", chem_class=ChemClass.SIGE_B, mode=Mode.BLANKET,
            T_K=_c_to_k(760.0), p_DCS=1.0, p_GeH4=r.GeH4_DCS, p_HCl=r.HCl_DCS,
            p_B2H6=r.B2H6_DCS, GR_nm_min=r.GR_nm_min, Ge_at_frac=r.Ge_at_pct / 100.0,
            source_dataset="DS2_GR",
        ))

    # ---- DS2: SiGe:B, [B] rows (unlinked to the GR rows above) --------------
    df2b = pd.read_csv(root / "tomasini_ds2_b.csv")
    for r in df2b.itertuples():
        rows.append(CanonicalRow(
            reactor_id="ASM_Epsilon", chem_class=ChemClass.SIGE_B, mode=Mode.BLANKET,
            T_K=_c_to_k(760.0), p_DCS=1.0, p_GeH4=r.GeH4_DCS, p_HCl=r.HCl_DCS,
            p_B2H6=r.B2H6_DCS, B_conc=r.B_conc_1e19_at_cm3 * 1e19,
            source_dataset="DS2_B",
        ))

    # ---- DS3: Hartmann, isothermal 750 C ------------------------------------
    df3 = pd.read_csv(root / "tomasini_ds3.csv")
    for r in df3.itertuples():
        rows.append(CanonicalRow(
            reactor_id="Hartmann", chem_class=ChemClass.SIGE, mode=Mode.BLANKET,
            T_K=_c_to_k(r.Tg_C), p_DCS=1.0,
            p_GeH4=r.GeH4_DCS_x10000 / 10000.0, p_HCl=r.HCl_DCS_x10000 / 10000.0,
            GR_nm_min=r.GR_nm_min, Ge_at_frac=r.Ge_at_pct / 100.0,
            source_dataset="DS3",
        ))

    # ---- DS4: Tan, absolute sccm flows -> DCS-normalized ratios ------------
    df4 = pd.read_csv(root / "tomasini_ds4.csv")
    for r in df4.itertuples():
        rows.append(CanonicalRow(
            reactor_id="Tan", chem_class=ChemClass.SIGE_B, mode=Mode.BLANKET,
            T_K=_c_to_k(r.Tg_C), p_DCS=1.0,
            p_GeH4=r.GeH4_sccm / r.DCS_sccm, p_HCl=r.HCl_sccm / r.DCS_sccm,
            p_B2H6=r.B2H6_sccm / r.DCS_sccm,
            GR_nm_min=None, Ge_at_frac=r.Ge_pct / 100.0,
            source_dataset="DS4",
        ))

    ds = Dataset(rows)
    ds.validate()
    return ds


# ---------------------------------------------------------------------------
# STANDARD INTAKE FORMAT for any data added AFTER the initial Tomasini
# reproduction (Phase 9+ "additive training": new epitaxy-team experiments,
# not a from-scratch retrain). Unlike ingest_tomasini (which has to match
# each appendix's own quirky column names), this defines ONE stable schema
# for all future data, Tomasini-shaped or not:
#
#   Legacy: T_C, HCl_over_DCS, GeH4_over_DCS, B2H6_over_DCS, GR_nm_min,
#   Ge_at_pct, B_conc_at_cm3
#
#   General: run_id, T_C, Si_source, Si_source_flow_sccm, HCl_over_Si,
#   GeH4_over_Si, MMS_over_Si, dopant_species, dopant_over_Si, raw flow
#   alternatives, GR/thickness/time, Ge_at_pct, C_at_pct, dopant targets
#
# growth_time_s is NOT required in this format because GR_nm_min is
# expected pre-computed -- but see the README's data-collection checklist:
# record raw growth time anyway, since Tomasini's own DS4 (no growth time
# in the appendix) is the one hard data gap this whole reproduction hit.
# ---------------------------------------------------------------------------
def ingest_standard_csv(path: str | Path, reactor_id: str, chem_class: ChemClass,
                        mode: Mode = Mode.BLANKET, source_tag: str = "") -> Dataset:
    """Ingest one CSV in the standard intake format above into canonical
    rows tagged with the GIVEN reactor_id/chem_class/source_tag (never
    inferred from the data -- the caller must state which reactor and
    chemistry class this data belongs to; see data_store.py for why that
    explicit tagging is exactly what keeps additive training from
    contaminating across reactors/classes)."""
    df = pd.read_csv(path)
    rows: list[CanonicalRow] = []
    for r in df.itertuples():
        public_class = parse_chem_class(chem_class)
        si_source = str(_get(r, "Si_source", "DCS"))
        si_flow = _get(r, "Si_source_flow_sccm", None)
        hcl_ratio = _ratio_or_flow(r, ("HCl_over_Si", "HCl_over_DCS"), "HCl_flow_sccm", si_flow)
        geh4_ratio = _ratio_or_flow(r, ("GeH4_over_Si", "GeH4_over_DCS"), "GeH4_flow_sccm", si_flow)
        mms_ratio = _ratio_or_flow(r, ("MMS_over_Si",), "MMS_flow_sccm", si_flow)
        dop_ratio = _ratio_or_flow(r, ("dopant_over_Si", "B2H6_over_DCS"), "dopant_flow_sccm", si_flow)
        b2h6_ratio = _get(r, "B2H6_over_DCS", None)
        if b2h6_ratio is None and _default_dopant_species(public_class, r).upper() == "B2H6":
            b2h6_ratio = dop_ratio
        if b2h6_ratio is None:
            b2h6_ratio = 0.0
        h2_ratio = _ratio_or_flow(r, ("H2_over_Si",), "H2_flow_sccm", si_flow)
        n2_ratio = _ratio_or_flow(r, ("N2_over_Si",), "N2_flow_sccm", si_flow)
        dopant_conc = _get(r, "dopant_conc_at_cm3", _get(r, "B_conc_at_cm3", None))
        b_conc = _get(r, "B_conc_at_cm3", None)
        if b_conc is None and _default_dopant_species(public_class, r).upper() == "B2H6":
            b_conc = dopant_conc
        ge_pct = _get(r, "Ge_at_pct", None)
        c_pct = _get(r, "C_at_pct", None)
        dop_pct = _get(r, "dopant_at_pct", None)
        rows.append(CanonicalRow(
            reactor_id=reactor_id, chem_class=public_class, mode=mode,
            T_K=_c_to_k(float(_get(r, "T_C"))), p_DCS=1.0, p_GeH4=geh4_ratio,
            p_HCl=hcl_ratio, p_B2H6=float(b2h6_ratio), p_MMS=mms_ratio,
            p_dopant=dop_ratio, p_H2=h2_ratio, p_N2=n2_ratio,
            p_carrier=h2_ratio + n2_ratio,
            XT_flow_H2_minus_N2_sccm=float(_get(r, "XT_flow_H2_minus_N2_sccm", 0.0) or 0.0),
            pattern_density=float(_get(r, "pattern_density", 0.0) or 0.0),
            run_id=str(_get(r, "run_id", "")),
            si_source=si_source,
            dopant_species=_default_dopant_species(public_class, r),
            growth_time_s=_get(r, "growth_time_s", None),
            GR_nm_min=_derive_gr_nm_min(r),
            Ge_at_frac=(float(ge_pct) / 100.0 if ge_pct is not None else None),
            C_at_frac=(float(c_pct) / 100.0 if c_pct is not None else None),
            B_conc=b_conc,
            dopant_conc=dopant_conc,
            dopant_at_frac=(float(dop_pct) / 100.0 if dop_pct is not None else None),
            source_dataset=source_tag or f"additional:{Path(path).stem}",
        ))
    ds = Dataset(rows)
    ds.validate()
    return ds
