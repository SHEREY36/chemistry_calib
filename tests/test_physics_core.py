"""Phase 3 tests: feature builder + the kappa(1/T) sign convention that the
whole reproduction hinges on (build_steps_and_cfd_integration.md Phase 3.1)."""
import jax.numpy as jnp
import numpy as np
import pytest

from chem_ml.config import Config
from chem_ml.features import build_features
from chem_ml.physics_core import destandardize_kappa, ge_logmodel, gr_logmodel
from chem_ml.schema import ingest_tomasini


@pytest.fixture(scope="module")
def ds1_features():
    ds = ingest_tomasini(Config().data_raw).filter(source_dataset="DS1")
    return build_features(ds), ds


def test_feature_columns_and_shape(ds1_features):
    fb, ds = ds1_features
    assert fb.col_names[:4] == ["invT", "ln_HCl", "ln_GeH4", "ln_B2H6"]
    assert fb.col_names[4:] == [
        "ln_C_source",
        "ln_dopant",
        "ln_H2",
        "ln_N2",
        "XT_H2_minus_N2_scaled",
        "pattern_density",
    ]
    assert fb.X.shape == (70, 10)
    # DS1 has no B2H6 -> the placeholder column must be exactly zero.
    assert jnp.all(fb.X[:, 3] == 0.0)
    assert jnp.all(fb.X[:, 4:] == 0.0)


def test_invT_standardization_round_trip(ds1_features):
    fb, ds = ds1_features
    mu, sd = fb.invT_scaler
    invT_raw = np.array([1.0 / r.T_K for r in ds.rows])
    invT_reconstructed = np.asarray(fb.X[:, 0]) * sd + mu
    np.testing.assert_allclose(invT_reconstructed, invT_raw, rtol=1e-10)


def test_destandardize_kappa_round_trip():
    invT_scaler = (2.5e-3, 3.0e-5)
    kappa_true_K = -24507.0
    _, sd = invT_scaler
    kappa_std = kappa_true_K * sd
    recovered = destandardize_kappa(kappa_std, invT_scaler)
    assert recovered == pytest.approx(kappa_true_K, rel=1e-10)


def test_gr_rises_with_temperature_requires_negative_kappa():
    """INVARIANT (physics_core.py docstring): kappa is the coefficient of
    1/T. Since GR empirically RISES with T, and 1/T FALLS as T rises, the
    coefficient must be NEGATIVE for the model to reproduce that trend. This
    test pins the sign so a future edit can't silently flip it."""
    params_correct_sign = {"lnK_GR": 0.0, "kappa_GR": -24507.0 * 3.0e-5,
                            "gamma_HCl": -0.7, "gamma_GeH4": 1.3}
    T_lo, T_hi = 900.0, 1000.0  # K; T_hi > T_lo
    mu, sd = 1.0 / 950.0, 3.0e-5
    invT_lo_s = (1.0 / T_lo - mu) / sd
    invT_hi_s = (1.0 / T_hi - mu) / sd
    X = jnp.array([[invT_lo_s, -0.5, -3.0, 0.0],
                   [invT_hi_s, -0.5, -3.0, 0.0]])
    ln_gr = gr_logmodel(params_correct_sign, X)
    gr_lo, gr_hi = float(jnp.exp(ln_gr[0])), float(jnp.exp(ln_gr[1]))
    assert gr_hi > gr_lo, "GR must increase with T given kappa_GR < 0"


def test_ge_falls_with_temperature_requires_positive_kappa():
    """Mirror of the GR test for the Ge/Si ratio: Ge fraction FALLS with T
    (confirmed against DS1: ~33% at 605 C vs ~21% at 765 C at matched
    GeH4/DCS), which -- given the IDENTICAL ln(y)=lnK+kappa/T functional
    form as GR -- requires kappa_Ge > 0, the OPPOSITE sign from kappa_GR.
    See physics_core.py module docstring for why this is not a typo."""
    params = {"lnK_Ge": 0.0, "kappa_Ge": 4319.0 * 3.0e-5,
              "dgamma_HCl": 0.1, "dgamma_GeH4": 0.51}
    T_lo, T_hi = 900.0, 1000.0
    mu, sd = 1.0 / 950.0, 3.0e-5
    invT_lo_s = (1.0 / T_lo - mu) / sd
    invT_hi_s = (1.0 / T_hi - mu) / sd
    X = jnp.array([[invT_lo_s, -0.5, -3.0, 0.0],
                   [invT_hi_s, -0.5, -3.0, 0.0]])
    ratio = jnp.exp(ge_logmodel(params, X))
    ratio_lo, ratio_hi = float(ratio[0]), float(ratio[1])
    assert ratio_hi < ratio_lo, "Ge/(1-Ge) must decrease with T given kappa_Ge < 0"


def test_legacy_gr_predictions_ignore_appended_features():
    params = {"lnK_GR": 1.0, "kappa_GR": -2.0, "gamma_HCl": -0.7, "gamma_GeH4": 1.3}
    X_legacy = jnp.array([[0.1, -0.3, -3.0, 0.0]])
    X_augmented = jnp.array([[0.1, -0.3, -3.0, 0.0, 8.0, -2.0, 4.0, 5.0, 1.1, 0.6]])

    assert jnp.array_equal(gr_logmodel(params, X_legacy), gr_logmodel(params, X_augmented))
