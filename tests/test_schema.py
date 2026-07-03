"""Phase 1 tests: unit conversion + fail-loud validation gates."""
import pytest

from chem_ml.config import Config
from chem_ml.pipeline import load_all_datasets
from chem_ml.schema import CanonicalRow, ChemClass, Dataset, Mode, ingest_tomasini


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
