"""Phase 1 tests: unit conversion + fail-loud validation gates."""
import pytest

from chem_ml.config import Config
from chem_ml.pipeline import load_all_datasets
from chem_ml.schema import (
    CanonicalRow,
    ChemClass,
    Dataset,
    Mode,
    canonical_chem_class,
    ingest_standard_csv,
    ingest_tomasini,
)


@pytest.fixture(scope="module")
def full_dataset():
    return ingest_tomasini(Config().data_raw)


def test_row_counts(full_dataset):
    assert len(full_dataset) == 152
    assert len(full_dataset.filter(source_dataset="DS1")) == 70
    assert len(full_dataset.filter(source_dataset="DS2_GR")) == 18
    assert len(full_dataset.filter(source_dataset="DS2_B")) == 11
    assert len(full_dataset.filter(source_dataset="DS3")) == 35
    assert len(full_dataset.filter(source_dataset="DS4")) == 18


def test_temperature_is_kelvin_not_celsius(full_dataset):
    """Every kinetics bug starts here (build doc, Phase 1.2): T_K must be the
    converted value, never the raw Celsius figure left unconverted."""
    for r in full_dataset.rows:
        assert r.T_K > 273.0
        # DS1 spans 605-765 C -> 878.15-1038.15 K; DS3 is fixed at 750 C -> 1023.15 K
        assert 800.0 < r.T_K < 1100.0

    ds1_765 = full_dataset.filter(source_dataset="DS1", T_K=765.0 + 273.15)
    assert len(ds1_765) > 0

    ds3 = full_dataset.filter(source_dataset="DS3")
    assert all(r.T_K == pytest.approx(750.0 + 273.15) for r in ds3.rows)


def test_validate_rejects_negative_pressure():
    bad = CanonicalRow(
        reactor_id="x", chem_class=ChemClass.SIGE, mode=Mode.BLANKET,
        T_K=1000.0, p_DCS=1.0, p_GeH4=-0.1, p_HCl=0.5,
    )
    with pytest.raises(AssertionError):
        bad.validate()


def test_validate_rejects_obviously_unconverted_celsius():
    """T_K <= 273 is unphysical for RPCVD growth and is exactly the failure
    mode of forgetting +273.15 on a low Celsius reading (e.g. a stray 25)."""
    bad = CanonicalRow(
        reactor_id="x", chem_class=ChemClass.SIGE, mode=Mode.BLANKET,
        T_K=25.0, p_DCS=1.0, p_GeH4=0.05, p_HCl=0.5,
    )
    with pytest.raises(AssertionError):
        bad.validate()


def test_validate_rejects_ge_fraction_out_of_range():
    bad = CanonicalRow(
        reactor_id="x", chem_class=ChemClass.SIGE, mode=Mode.BLANKET,
        T_K=1000.0, p_DCS=1.0, p_GeH4=0.05, p_HCl=0.5, Ge_at_frac=1.5,
    )
    with pytest.raises(AssertionError):
        bad.validate()


def test_ds4_has_no_growth_rate(full_dataset):
    """Known data gap (build_steps_and_cfd_integration.md Phase 1, DS4 note):
    Appendix III gives no growth time, so GR cannot be computed."""
    ds4 = full_dataset.filter(source_dataset="DS4")
    assert all(r.GR_nm_min is None for r in ds4.rows)
    assert all(r.Ge_at_frac is not None for r in ds4.rows)


def test_load_all_datasets_pipeline_matches_direct_ingest(full_dataset):
    ds = load_all_datasets(Config())
    assert len(ds) == len(full_dataset)


def test_legacy_standard_csv_intake_still_maps_dcs_ratios(tmp_path):
    csv_path = tmp_path / "legacy.csv"
    csv_path.write_text(
        "T_C,HCl_over_DCS,GeH4_over_DCS,B2H6_over_DCS,GR_nm_min,Ge_at_pct,B_conc_at_cm3\n"
        "730,0.6,0.03,0.0002,40.0,20.0,1.2e19\n"
    )

    row = ingest_standard_csv(
        csv_path,
        reactor_id="ASM_Epsilon",
        chem_class=ChemClass.SIGE_B,
        mode=Mode.BLANKET,
        source_tag="legacy",
    ).rows[0]

    assert row.si_source == "DCS"
    assert row.p_HCl == pytest.approx(0.6)
    assert row.p_GeH4 == pytest.approx(0.03)
    assert row.p_B2H6 == pytest.approx(0.0002)
    assert row.p_dopant == pytest.approx(0.0002)
    assert row.B_conc == pytest.approx(1.2e19)
    assert canonical_chem_class(row.chem_class) == ChemClass.SIGE_X


def test_sigec_raw_flow_intake_computes_source_normalized_ratios(tmp_path):
    csv_path = tmp_path / "sigec_flows.csv"
    csv_path.write_text(
        "run_id,T_C,Si_source,Si_source_flow_sccm,GeH4_flow_sccm,HCl_flow_sccm,"
        "MMS_flow_sccm,H2_flow_sccm,N2_flow_sccm,XT_flow_H2_minus_N2_sccm,"
        "growth_time_s,thickness_nm,Ge_at_pct,C_at_pct\n"
        "s1,700,SiH4,100,4,20,0.5,5000,1000,4000,60,120,22,0.8\n"
    )

    row = ingest_standard_csv(
        csv_path,
        reactor_id="XYZ_tool_1",
        chem_class=ChemClass.SIGEC,
        source_tag="sigec",
    ).rows[0]

    assert row.run_id == "s1"
    assert row.si_source == "SiH4"
    assert row.p_GeH4 == pytest.approx(0.04)
    assert row.p_HCl == pytest.approx(0.2)
    assert row.p_MMS == pytest.approx(0.005)
    assert row.p_H2 == pytest.approx(50.0)
    assert row.p_N2 == pytest.approx(10.0)
    assert row.XT_flow_H2_minus_N2_sccm == pytest.approx(4000.0)
    assert row.GR_nm_min == pytest.approx(120.0)
    assert row.Ge_at_frac == pytest.approx(0.22)
    assert row.C_at_frac == pytest.approx(0.008)


def test_sigec_ratio_form_intake_without_flow_columns(tmp_path):
    csv_path = tmp_path / "sigec_ratios.csv"
    csv_path.write_text(
        "run_id,T_C,Si_source,HCl_over_Si,GeH4_over_Si,MMS_over_Si,GR_nm_min,Ge_at_pct,C_at_pct\n"
        "s2,710,trisilane,0.18,0.035,0.004,95,24,0.7\n"
    )

    row = ingest_standard_csv(
        csv_path,
        reactor_id="XYZ_tool_1",
        chem_class=ChemClass.SIGEC,
        source_tag="sigec_ratio",
    ).rows[0]

    assert row.si_source == "trisilane"
    assert row.p_HCl == pytest.approx(0.18)
    assert row.p_GeH4 == pytest.approx(0.035)
    assert row.p_MMS == pytest.approx(0.004)


def test_sigecx_preserves_generic_dopant_fields(tmp_path):
    csv_path = tmp_path / "sigecx.csv"
    csv_path.write_text(
        "run_id,T_C,Si_source,HCl_over_Si,GeH4_over_Si,MMS_over_Si,dopant_species,"
        "dopant_over_Si,GR_nm_min,Ge_at_pct,C_at_pct,dopant_conc_at_cm3,dopant_at_pct\n"
        "sx1,715,DCS,0.2,0.03,0.003,PH3,0.0001,100,23,0.6,3e19,0.02\n"
    )

    row = ingest_standard_csv(
        csv_path,
        reactor_id="XYZ_tool_1",
        chem_class=ChemClass.SIGEC_X,
        source_tag="sigecx",
    ).rows[0]

    assert row.chem_class == ChemClass.SIGEC_X
    assert row.dopant_species == "PH3"
    assert row.p_dopant == pytest.approx(0.0001)
    assert row.dopant_conc == pytest.approx(3e19)
    assert row.dopant_at_frac == pytest.approx(0.0002)


def test_flow_columns_require_si_source_flow_denominator(tmp_path):
    csv_path = tmp_path / "bad_flows.csv"
    csv_path.write_text(
        "T_C,Si_source,HCl_flow_sccm,GR_nm_min\n"
        "700,SiH4,20,100\n"
    )

    with pytest.raises(ValueError, match="Si_source_flow_sccm"):
        ingest_standard_csv(csv_path, reactor_id="XYZ_tool_1", chem_class=ChemClass.SIGEC)
