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
    SIGE = "SiGe"
    SIGE_B = "SiGe:B"
    SIGE_P = "SiGe:P"
    SIC = "SiC"
    SIGEC = "SiGeC"


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
    p_carrier: float = 0.0
    pattern_density: float = 0.0          # 0 for blanket
    growth_time_s: Optional[float] = None
    GR_nm_min: Optional[float] = None
    Ge_at_frac: Optional[float] = None    # 0..1
    B_conc: Optional[float] = None        # at/cm^3
    source_dataset: str = ""

    def validate(self) -> None:
        """Fail-loud validation (Phase 1.3). Raise on physical impossibility."""
        assert self.T_K > 273.0, f"T_K={self.T_K} looks like Celsius; convert at ingest"
        for name in ("p_DCS", "p_GeH4", "p_HCl", "p_B2H6"):
            v = getattr(self, name)
            assert v >= 0.0, f"{name} negative ({v})"
        assert self.p_DCS > 0.0, "p_DCS must be > 0 (normalization base)"
        if self.GR_nm_min is not None:
            assert self.GR_nm_min > 0.0, "GR must be > 0"
        if self.Ge_at_frac is not None:
            assert 0.0 < self.Ge_at_frac < 1.0, "Ge fraction out of (0,1)"


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

    def __len__(self) -> int:
        return len(self.rows)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.rows])


def _c_to_k(t_c: float) -> float:
    return t_c + 273.15


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
