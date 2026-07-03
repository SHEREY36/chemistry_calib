"""
Phase 4/6/7 validation report generator: runs the full Phases 1-8 pipeline
and renders a markdown report placing every reproduced number next to the
paper's own published value, so "we reproduced Tomasini" is an artifact,
not a claim (build_steps_and_cfd_integration.md "Testing & verification").
"""
from __future__ import annotations

import datetime
import platform

from chem_ml.config import Config
from chem_ml.pipeline import (
    load_all_datasets,
    run_phase4_calibration,
    run_phase6_identifiability,
    run_phase7_cross_reactor,
    run_phase8_inverse_design,
)


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def generate_validation_report(cfg: Config | None = None) -> str:
    cfg = cfg or Config()
    ds = load_all_datasets(cfg)
    p4 = run_phase4_calibration(cfg, ds)
    p6 = run_phase6_identifiability(cfg, p4)
    p7 = run_phase7_cross_reactor(cfg, p4, ds)
    p8_inrange = run_phase8_inverse_design(cfg, p4, target_gr_nm_min=29.3, target_ge_frac=0.2173)
    p8_extreme = run_phase8_inverse_design(cfg, p4, target_gr_nm_min=500.0, target_ge_frac=0.60)

    r4, r6, r7 = p4["report"], p6["report"], p7["report"]
    lines: list[str] = []
    w = lines.append

    w("# Tomasini et al. (2010) Reproduction: Validation Report")
    w("")
    w(f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} "
      f"on Python {platform.python_version()}.")
    w("")
    w("Source: P. Tomasini, V. Machkaoutsan, S.G. Thomas, \"Analysis of silicon "
      "germanium vapor phase epitaxy kinetics,\" *Thin Solid Films* 518 (2010) S12-S17.")
    w("")
    w(f"Dataset: {len(ds)} canonical rows ingested from the paper's appendices "
      f"(DS1={len(ds.filter(source_dataset='DS1'))}, "
      f"DS2_GR={len(ds.filter(source_dataset='DS2_GR'))}, "
      f"DS2_B={len(ds.filter(source_dataset='DS2_B'))}, "
      f"DS3={len(ds.filter(source_dataset='DS3'))}, "
      f"DS4={len(ds.filter(source_dataset='DS4'))}).")
    w("")

    w("## Phase 4 -- Bayesian calibration on DS1/DS2 (the core reproduction)")
    w("")
    w("| Metric | This pipeline | Paper | Target |")
    w("|---|---|---|---|")
    w(f"| DS1 GR parity R² | {_fmt(r4['R2_GR'])} | 0.985 (Eq. 10) | ≥ 0.98 |")
    w(f"| DS1 Ge% parity R² | {_fmt(r4['R2_Ge'])} | 0.988 (Eq. 15) | ≥ 0.98 |")
    w(f"| κ_GR (= paper's Ea/R) | {_fmt(r4['kappa_GR_K'], 0)} K | -24,507 K | within ±10% |")
    w(f"| γ_HCl | {_fmt(r4['gamma_HCl'])} | -0.7 | ±0.1 |")
    w(f"| γ_GeH4 | {_fmt(r4['gamma_GeH4'])} | 1.3 | ±0.15 |")
    w(f"| DS2 [B] parity R² | {_fmt(r4['R2_B'])} | -- (Eq. 20) | -- |")
    w(f"| β_B2H6 | {_fmt(r4['beta_B2H6'])} | ~0.8 (Eq. 20) | ±0.2 |")
    w(f"| **Overall PASS** | **{r4['PASS']}** | | |")
    w("")
    w("Note: `kappa_Ge`'s sign was corrected during implementation (paper's tabulated "
      "\"ΔEa/R = -4319\" does not transfer directly to this parametrization -- see "
      "`chem_ml/physics_core.py` docstring). Verified directly against DS1: at matched "
      "GeH4/DCS≈0.045, Ge% is ~33% at 605 C vs ~21% at 765 C.")
    w("")

    w("## Phase 6 -- Sensitivity analysis (reproduces Fig. 4)")
    w("")
    w("Operating points from Fig. 4's own caption: pHCl/pDCS=0.34 fixed, pGeH4/pDCS "
      "solved per-T to hit 20% Ge. These exact points were not part of the Phase 4 fit "
      "target, so recovering them is an independent check.")
    w("")
    w("| T (C) | GR (this pipeline, nm/min) | GR (paper, nm/min) |")
    w("|---|---|---|")
    for row in r6["sensitivity_table"]:
        w(f"| {row['T_C']:.0f} | {_fmt(row['GR_model_nm_min'])} | {_fmt(row['GR_paper_nm_min'])} |")
    w("")
    w(f"dGR/dT at 750 C: **{_fmt(r6['dGR_dT_at_750C'])} nm/min/K** "
      f"(paper: 1-2 nm/min/K range) -- {'within range' if r6['dGR_dT_in_paper_1_to_2_range'] else 'OUT OF RANGE'}.")
    w("")
    w(f"Posterior eigenspectrum (GR model, 4 params): stiffest = `{r6['stiffest_param']}`, "
      f"sloppiest = `{r6['sloppiest_param']}` "
      f"(eigenvalue ratio {r6['eigvals_ascending'][-1] / r6['eigvals_ascending'][0]:.0f}x).")
    w("")

    w("## Phase 7 -- Cross-reactor validation (DS3 Hartmann, DS4 Tan)")
    w("")
    w("theta_chem frozen at its Phase 4 DS1 posterior mean; only a "
      "3-4 parameter delta_r fit per reactor (no chemistry refit).")
    w("")
    w("| Dataset | Observable | This pipeline R² | Paper R² | delta_r params |")
    w("|---|---|---|---|---|")
    w(f"| DS3 (Hartmann) | GR | {_fmt(r7['DS3_R2_GR'])} | {_fmt(r7['DS3_R2_GR_paper'])} (Eq. 11) | {r7['DS3_n_delta_r_params']} |")
    w(f"| DS3 (Hartmann) | Ge/Si | {_fmt(r7['DS3_R2_Ge'])} | {_fmt(r7['DS3_R2_Ge_paper'])} (Eq. 16) | {r7['DS3_n_delta_r_params']} |")
    w(f"| DS4 (Tan) | Ge/Si | {_fmt(r7['DS4_R2_Ge'])} | ~{_fmt(r7['DS4_R2_Ge_paper'])} (Eqs. 18-19, blended) | {r7['DS4_n_delta_r_params']} |")
    w(f"| **Overall PASS** | | **{r7['PASS']}** | | |")
    w("")
    w("DS3's own paper GR R² (0.844) is already well below DS1's (0.985) -- that's the "
      "dataset Fig. 1's Regime-I curvature comes from, so matching it (not beating it) is "
      "the correct reproduction. DS4's Ge/Si gap vs. the paper is a documented consequence "
      "of theta_ge being frozen boron-free from DS1, applied to DS4 which has a trace-B2H6 "
      "term the paper's own DS4-specific model includes and this one deliberately doesn't "
      "(see `chem_ml/reactor_transfer.py` docstring).")
    w("")

    w("## Phase 8 -- Inverse design (spot check)")
    w("")
    w("| Target | Achieved GR | Achieved Ge | Confidence |")
    w("|---|---|---|---|")
    w(f"| GR=29.3 nm/min, Ge=21.7% (matches DS1 run #70) | {_fmt(p8_inrange['achieved_gr_nm_min'])} nm/min "
      f"({_fmt(100*p8_inrange['gr_rel_error'])}% err) | {_fmt(100*p8_inrange['achieved_ge_frac'])}% | "
      f"{'ACCEPTED' if p8_inrange['accepted'] else 'REFUSED'} |")
    w(f"| GR=500 nm/min, Ge=60% (outside DS1's range) | {_fmt(p8_extreme['achieved_gr_nm_min'])} nm/min "
      f"({_fmt(100*p8_extreme['gr_rel_error'])}% err) | {_fmt(100*p8_extreme['achieved_ge_frac'])}% | "
      f"{'ACCEPTED' if p8_extreme['accepted'] else 'REFUSED'} |")
    w("")

    w("## Known data gaps (documented, not silently worked around)")
    w("")
    w("- **DS4 has no growth time** in Tomasini's Appendix III, so GR cannot be computed "
      "for DS4 (only Thickness/Ge% are usable) -- DS4 is Ge/Si-only throughout this pipeline.")
    w("- **No sccm<->ratio conversion** is available for DS1, so Phase 6's dxGe/dpGeH4 is "
      "reported in per-unit-ratio terms only, not directly comparable to Fig. 5's per-sccm number.")
    w("")

    all_pass = r4["PASS"] and r7["PASS"] and r6["dGR_dT_in_paper_1_to_2_range"]
    w(f"## Summary: {'ALL GATES PASS' if all_pass else 'SOME GATES FAILED -- see above'}")
    w("")

    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_validation_report()
    with open("VALIDATION_REPORT.md", "w") as f:
        f.write(report)
    print(report)
