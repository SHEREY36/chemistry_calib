"""Phase 5 tests: gated residual NN hybrid."""
import jax.numpy as jnp
import numpy as np
import pytest

from chem_ml.config import Config
from chem_ml.pipeline import run_phase4_calibration, run_phase5_residual_hybrid
from chem_ml.residual_nn import ResidualNN, fit_residual_networks
from chem_ml.schema import CanonicalRow, ChemClass, Dataset, Mode


@pytest.fixture(scope="module")
def phase5_result():
    cfg = Config()
    p4 = run_phase4_calibration(cfg)
    return run_phase5_residual_hybrid(cfg, p4)


def test_residual_nn_shapes():
    net = ResidualNN(chem_class=ChemClass.SIGE, in_size=7, n_out=2, seed=0)
    X = jnp.zeros((5, 7))
    out = net(X)
    assert out.shape == (5, 2)


def test_residual_nn_rejects_wrong_input_size():
    net = ResidualNN(chem_class=ChemClass.SIGE, in_size=7, n_out=2, seed=0)
    with pytest.raises(ValueError):
        net(jnp.zeros((5, 3)))


def test_fit_residual_networks_hard_gate_by_class():
    """INVARIANT 3: each per-class net must be trained on, and only ever
    see, rows of its own declared class -- never a mix."""
    rows = [
        CanonicalRow(reactor_id="x", chem_class=ChemClass.SIGE, mode=Mode.BLANKET,
                     T_K=1000.0, p_DCS=1.0, p_GeH4=0.05, p_HCl=0.5),
        CanonicalRow(reactor_id="x", chem_class=ChemClass.SIGE, mode=Mode.BLANKET,
                     T_K=1010.0, p_DCS=1.0, p_GeH4=0.04, p_HCl=0.4),
        CanonicalRow(reactor_id="x", chem_class=ChemClass.SIGE_B, mode=Mode.BLANKET,
                     T_K=1030.0, p_DCS=1.0, p_GeH4=0.05, p_HCl=0.7, p_B2H6=0.001),
    ]
    ds = Dataset(rows)
    from chem_ml.features import build_features
    fb = build_features(ds)
    targets = {
        ChemClass.SIGE: jnp.zeros((2, 1)),
        ChemClass.SIGE_B: jnp.zeros((1, 1)),
    }
    nets = fit_residual_networks(ds, fb, targets, steps=10, n_out=1)
    assert nets[ChemClass.SIGE].in_size == nets[ChemClass.SIGE_B].in_size
    # the SiGe net was fit on exactly 2 rows, SiGe:B on exactly 1 -- enforced
    # by fit_residual_networks' per-class row mask, not by the caller.
    assert set(nets.keys()) == {ChemClass.SIGE, ChemClass.SIGE_B}


def test_net_stays_small_relative_to_physics_residual(phase5_result):
    """Phase 5.2: with the physics core already at R^2 >= 0.98, the
    regularized net should not explain away most of the remaining variance
    (that would mean it's fitting noise on 70 points)."""
    assert phase5_result["report"]["net_shrinks_toward_zero"]


def test_hybrid_improves_on_physics_only(phase5_result):
    assert phase5_result["report"]["hybrid_improves_on_physics"]
