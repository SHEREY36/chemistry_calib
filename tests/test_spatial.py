"""Phase 12 tests: spatial wafer-scan data model, ingestion, registration,
and the radially-resolved reactor-transfer fit (spatial.py, spatial_ingest.py,
data_store.py's wafer-scan registration, pipeline.run_phase12_spatial_transfer).

Real XYZ-reactor scan files aren't in-hand yet (per the plan), so every test
here builds synthetic fixtures, same pattern as tests/test_data_store.py."""
import numpy as np
import pytest

from chem_ml.config import Config
from chem_ml.pipeline import run_phase4_calibration, run_phase12_spatial_transfer
from chem_ml.schema import ChemClass
from chem_ml.spatial import (
    WaferPoint, WaferRunMeta, WaferScan, radial_profile, wafer_average, wiwnu,
)


@pytest.fixture(scope="module")
def phase4():
    return run_phase4_calibration(Config())


# ---------------------------------------------------------------------------
# WaferPoint: x/y -> r/theta derivation, validation
# ---------------------------------------------------------------------------
def test_waferpoint_derives_r_and_theta_from_xy():
    p = WaferPoint(run_id="r1", x_mm=3.0, y_mm=4.0, GR_nm_min_local=10.0)
    assert p.r_mm == pytest.approx(5.0)
    assert p.theta_deg == pytest.approx(53.13, abs=0.1)


def test_waferpoint_accepts_r_mm_directly_for_pure_radial_scans():
    p = WaferPoint(run_id="r1", x_mm=100.0, y_mm=0.0, thickness_A_local=500.0)
    assert p.r_mm == pytest.approx(100.0)
    assert p.theta_deg == pytest.approx(0.0)


def test_waferpoint_validate_requires_at_least_one_local_field():
    p = WaferPoint(run_id="r1", x_mm=10.0, y_mm=0.0)
    with pytest.raises(AssertionError):
        p.validate()


# ---------------------------------------------------------------------------
# Radial-profile utilities: hand-computed against a small synthetic profile
# ---------------------------------------------------------------------------
def test_radial_profile_averages_azimuthal_duplicates():
    r_mm = np.array([0.0, 0.0, 10.0, 10.0, 20.0])
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    r_p, v_p = radial_profile(r_mm, values)
    assert r_p.tolist() == [0.0, 10.0, 20.0]
    assert v_p.tolist() == pytest.approx([2.0, 6.0, 9.0])


def test_wafer_average_matches_hand_computed_trapz():
    r_mm = np.array([0.0, 0.0, 10.0, 10.0, 20.0])
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    # radial profile: r=[0,10,20], v=[2,6,9] -> (2/R_w^2) * trapz(r*v, r), R_w=20
    assert wafer_average(r_mm, values) == pytest.approx(7.5)


def test_wiwnu_matches_hand_computed_definition():
    r_mm = np.array([0.0, 0.0, 10.0, 10.0, 20.0])
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    # profile v=[2,6,9] -> (9-2)/2 / mean([2,6,9]) = 3.5 / (17/3)
    assert wiwnu(r_mm, values) == pytest.approx(3.5 / (17.0 / 3.0))


# ---------------------------------------------------------------------------
# WaferScan.to_canonical_row / effective_GR_nm_min
# ---------------------------------------------------------------------------
def _make_scan(run_id="scan1", reactor_id="XYZ_tool_1", growth_time_s=None):
    meta = WaferRunMeta(
        run_id=run_id, reactor_id=reactor_id, chem_class=ChemClass.SIGE,
        T_set_K=1000.0, p_DCS=1.0, p_GeH4=0.03, p_HCl=0.5, growth_time_s=growth_time_s,
    )
    points = [
        WaferPoint(run_id=run_id, x_mm=0.0, y_mm=0.0, GR_nm_min_local=40.0, Ge_at_frac_local=0.22),
        WaferPoint(run_id=run_id, x_mm=100.0, y_mm=0.0, GR_nm_min_local=36.0, Ge_at_frac_local=0.20),
        WaferPoint(run_id=run_id, x_mm=150.0, y_mm=0.0, GR_nm_min_local=32.0, Ge_at_frac_local=0.18),
    ]
    return WaferScan(meta=meta, points=points)


def test_to_canonical_row_center_reduction_uses_nearest_point():
    scan = _make_scan()
    row = scan.to_canonical_row(reduction="center")
    assert row.GR_nm_min == pytest.approx(40.0)
    assert row.Ge_at_frac == pytest.approx(0.22)
    assert row.reactor_id == "XYZ_tool_1"
    assert row.chem_class == ChemClass.SIGE


def test_to_canonical_row_area_weighted_mean_uses_wafer_average():
    scan = _make_scan()
    row = scan.to_canonical_row(reduction="area_weighted_mean")
    r_mm = scan.r_array()
    gr_vals = np.array([p.GR_nm_min_local for p in scan.points])
    assert row.GR_nm_min == pytest.approx(wafer_average(r_mm, gr_vals))


def test_effective_gr_derived_from_thickness_and_growth_time():
    scan = _make_scan(growth_time_s=600.0)  # 10 min
    pt = WaferPoint(run_id=scan.meta.run_id, x_mm=50.0, y_mm=0.0, thickness_A_local=1000.0)
    # thickness_A/10 -> 100 nm; growth_time_s/60 -> 10 min; GR = 10 nm/min
    assert scan.effective_GR_nm_min(pt) == pytest.approx(10.0)


def test_effective_gr_none_without_thickness_or_growth_time():
    scan = _make_scan(growth_time_s=None)
    pt = WaferPoint(run_id=scan.meta.run_id, x_mm=50.0, y_mm=0.0, thickness_A_local=1000.0)
    assert scan.effective_GR_nm_min(pt) is None


# ---------------------------------------------------------------------------
# Ingestion: synthetic runs.csv / points.csv, Stick_i / probe_i columns
# ---------------------------------------------------------------------------
def test_ingest_wafer_scan_csv_parses_runs_and_points(tmp_path):
    from chem_ml.spatial_ingest import ingest_wafer_scan_csv

    runs_csv = tmp_path / "runs.csv"
    runs_csv.write_text(
        "run_id,T_set_C,HCl_over_DCS,GeH4_over_DCS,growth_time_s,Stick_1,Stick_2,probe_1\n"
        "wafer_1,727.0,0.5,0.03,600,50.0,25.0,730.0\n"
    )
    points_csv = tmp_path / "points.csv"
    points_csv.write_text(
        "run_id,x_mm,y_mm,GR_nm_min_local,Ge_at_pct_local,measurement_source\n"
        "wafer_1,0,0,40.0,22.0,SE_contour\n"
        "wafer_1,147,0,32.0,18.0,SE_contour\n"
        "wafer_1,104,104,33.0,18.5,SE_contour\n"
    )

    scans = ingest_wafer_scan_csv(str(runs_csv), str(points_csv), reactor_id="XYZ_tool_1",
                                  chem_class=ChemClass.SIGE, source_tag="test_scan")
    assert len(scans) == 1
    scan = scans[0]
    assert len(scan.points) == 3
    assert scan.meta.nozzle_flows_sccm == {"Stick_1": 50.0, "Stick_2": 25.0}
    assert scan.meta.probe_temps_K["probe_1"] == pytest.approx(730.0 + 273.15)
    assert scan.meta.T_set_K == pytest.approx(727.0 + 273.15)
    center = next(p for p in scan.points if p.r_mm == pytest.approx(0.0))
    assert center.Ge_at_frac_local == pytest.approx(0.22)


# ---------------------------------------------------------------------------
# Registration: dedup-by-source_tag, same guarantee as register_new_data
# ---------------------------------------------------------------------------
def _write_scan_csvs(tmp_path, run_id="wafer_1"):
    runs_csv = tmp_path / "runs.csv"
    runs_csv.write_text(
        "run_id,T_set_C,HCl_over_DCS,GeH4_over_DCS\n"
        f"{run_id},727.0,0.5,0.03\n"
    )
    points_csv = tmp_path / "points.csv"
    points_csv.write_text(
        "run_id,x_mm,y_mm,GR_nm_min_local,Ge_at_pct_local\n"
        f"{run_id},0,0,40.0,22.0\n"
        f"{run_id},100,0,36.0,20.0\n"
    )
    return runs_csv, points_csv


def test_register_and_load_wafer_scan_roundtrip(tmp_path):
    from chem_ml.data_store import load_registered_wafer_scans, register_wafer_scan, registered_spatial_tags

    cfg = Config(data_raw="data/raw", data_processed=str(tmp_path))
    runs_csv, points_csv = _write_scan_csvs(tmp_path)

    register_wafer_scan(cfg, str(runs_csv), str(points_csv), reactor_id="XYZ_tool_1",
                        chem_class=ChemClass.SIGE, source_tag="scan_batch_1")
    assert "scan_batch_1" in registered_spatial_tags(cfg)

    scans = load_registered_wafer_scans(cfg, "scan_batch_1")
    assert len(scans) == 1
    assert len(scans[0].points) == 2


def test_reregistering_same_scan_tag_raises(tmp_path):
    from chem_ml.data_store import register_wafer_scan

    cfg = Config(data_raw="data/raw", data_processed=str(tmp_path))
    runs_csv, points_csv = _write_scan_csvs(tmp_path)
    register_wafer_scan(cfg, str(runs_csv), str(points_csv), reactor_id="XYZ_tool_1",
                        chem_class=ChemClass.SIGE, source_tag="dup_scan_tag")
    with pytest.raises(ValueError, match="already registered"):
        register_wafer_scan(cfg, str(runs_csv), str(points_csv), reactor_id="XYZ_tool_1",
                            chem_class=ChemClass.SIGE, source_tag="dup_scan_tag")


# ---------------------------------------------------------------------------
# The actual deliverable: radially-resolved reactor-transfer recovers a
# PLANTED linear radial trend in the delivered HCl ratio.
# ---------------------------------------------------------------------------
def test_spatial_transfer_recovers_planted_radial_alpha_trend(phase4):
    import jax.numpy as jnp
    from chem_ml.calibration import posterior_mean_params
    from chem_ml.physics_core import ge_logmodel, gr_logmodel

    theta_gr = posterior_mean_params(phase4["mcmc_gr"], ["lnK_GR", "kappa_GR", "gamma_HCl", "gamma_GeH4"])
    theta_ge = posterior_mean_params(phase4["mcmc_ge"], ["lnK_Ge", "kappa_Ge", "dgamma_HCl", "dgamma_GeH4"])
    mu, sd = phase4["features_ds1"].invT_scaler

    T_K = 1000.0
    hcl_ratio, geh4_ratio = 0.5, 0.035
    invT_std = (1.0 / T_K - mu) / sd
    R_w = 150.0
    r_values = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
    a1_true = 0.4  # planted: delivered HCl ratio effectively rises toward the wafer edge

    # NOTE: Ge/Si's own HCl order (dgamma_HCl ~0.1, METHODOLOGY.md sec 5) is
    # much weaker than GR's (gamma_HCl ~-0.7) -- so the Ge channel's signal
    # from this same planted alpha_HCl(r) trend is intrinsically much
    # smaller than GR's. Use a small noise level so this stays a clean
    # architecture-recovery check, not a noise-robustness stress test.
    rng = np.random.default_rng(0)
    points = []
    for r in r_values:
        ln_alpha_HCl_r = a1_true * (r / R_w)
        X = jnp.array([[invT_std, jnp.log(hcl_ratio) + ln_alpha_HCl_r, jnp.log(geh4_ratio), 0.0]])
        gr_true = float(jnp.exp(gr_logmodel(theta_gr, X))[0]) * np.exp(rng.normal(0, 0.005))
        ge_ratio_true = float(jnp.exp(ge_logmodel(theta_ge, X))[0]) * np.exp(rng.normal(0, 0.005))
        ge_true = ge_ratio_true / (1 + ge_ratio_true)
        points.append(WaferPoint(run_id="scan1", x_mm=float(r), y_mm=0.0,
                                 GR_nm_min_local=gr_true, Ge_at_frac_local=ge_true))

    meta = WaferRunMeta(run_id="scan1", reactor_id="XYZ_tool_1", chem_class=ChemClass.SIGE,
                        T_set_K=T_K, p_DCS=1.0, p_GeH4=geh4_ratio, p_HCl=hcl_ratio)
    scan = WaferScan(meta=meta, points=points)
    scan.validate()

    out = run_phase12_spatial_transfer(Config(), phase4, scan)
    assert out["diag"]["max_rhat"] < 1.05
    assert out["report"]["a1_alpha_HCl"] > 0  # recovers the PLANTED direction
    assert out["report"]["R2_GR"] > 0.9
    assert out["report"]["R2_Ge"] > 0.9
