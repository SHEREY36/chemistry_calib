"""
Phase 8 demo: inverse design against the calibrated Tomasini reproduction.

Given a target (GR, %Ge), find the recipe (T, pHCl/pDCS, pGeH4/pDCS) that
achieves it, with a confidence flag that REFUSES targets the model has no
business extrapolating to.

This is a plain script rather than a .ipynb: everything it calls
(run_phase4_calibration, run_phase8_inverse_design) is already covered by
pytest in tests/test_inverse_design.py, so the script's job is purely
demonstrative -- run it and read the printed recipes, don't treat it as the
source of truth for correctness.

Run: conda activate epitaxy && python notebooks/phase8_inverse_design_demo.py
"""
import logging

from chem_ml.config import Config
from chem_ml.pipeline import run_phase4_calibration, run_phase8_inverse_design

logging.basicConfig(level=logging.WARNING)


def _print_result(label: str, r: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  target:   GR={r['target_gr_nm_min']:.1f} nm/min, Ge={100*r['target_ge_frac']:.1f}%")
    print(f"  achieved: GR={r['achieved_gr_nm_min']:.1f} nm/min "
          f"({100*r['gr_rel_error']:.1f}% rel. error), "
          f"Ge={100*r['achieved_ge_frac']:.1f}% ({100*r['ge_abs_error']:.1f} pt abs. error)")
    rec = r["recipe"]
    print(f"  recipe:   T={rec['T_K']-273.15:.0f} C, "
          f"pHCl/pDCS={rec['p_HCl_over_pDCS']:.4f}, pGeH4/pDCS={rec['p_GeH4_over_pDCS']:.4f}")
    print(f"  confidence: {'ACCEPTED' if r['accepted'] else 'REFUSED (low confidence)'} "
          f"(UQ={r['uq_at_solution']:.5f} vs baseline={r['baseline_uq']:.5f}, "
          f"at_feasible_boundary={r['at_feasible_boundary']})")


def main() -> None:
    cfg = Config()
    print("Fitting Phase 4 posterior (this is the frozen theta_chem inverse design optimizes against)...")
    p4 = run_phase4_calibration(cfg)

    # (1) In-range: matches DS1 run #70 (Tg=725C, GR=29.3 nm/min, Ge=21.73%)
    # almost exactly -- should recover the recipe with small error.
    r1 = run_phase8_inverse_design(cfg, p4, target_gr_nm_min=29.3, target_ge_frac=0.2173)
    _print_result("In-range target (matches DS1 run #70)", r1)

    # (2) A practical mid-range target not equal to any single DS1 row.
    r2 = run_phase8_inverse_design(cfg, p4, target_gr_nm_min=50.0, target_ge_frac=0.25)
    _print_result("Practical target (GR=50 nm/min, Ge=25%)", r2)

    # (3) Extreme/infeasible: GR=500 nm/min at Ge=60% simultaneously -- far
    # outside anything DS1 observed (max GR=128 nm/min, max Ge=40.9%).
    # Should be refused, not silently extrapolated.
    r3 = run_phase8_inverse_design(cfg, p4, target_gr_nm_min=500.0, target_ge_frac=0.60)
    _print_result("Extreme/infeasible target (GR=500 nm/min, Ge=60%)", r3)


if __name__ == "__main__":
    main()
