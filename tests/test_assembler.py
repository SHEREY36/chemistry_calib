"""Phase 2 tests: registry/assembler, and the anti-contamination guarantee
(build_steps_and_cfd_integration.md Phase 2.3) -- absent species must
contribute EXACTLY zero, not "a small learned amount".
"""
import jax.numpy as jnp
import pytest

from chem_ml.assembler import ReactionNetworkAssembler
from chem_ml.physics_core import gr_logmodel
from chem_ml.registry import Role, SpeciesRegistry
from chem_ml.schema import ChemClass, Mode


@pytest.fixture
def reg():
    return SpeciesRegistry()


@pytest.fixture
def asm(reg):
    return ReactionNetworkAssembler(reg)


def test_registry_seeds_expected_species(reg):
    for name in ("dichlorosilane", "germane", "methylsilane", "hcl", "diborane", "hydrogen", "nitrogen"):
        assert reg.get(name).canonical_name == name
    with pytest.raises(KeyError):
        reg.get("unobtainium")


def test_registry_add_rejects_duplicates(reg):
    with pytest.raises(ValueError):
        from chem_ml.registry import Species
        reg.add(Species("germane", "GeH4", Role.GE_SOURCE, "germane"))


def test_assemble_sige_no_dopant(asm):
    net = asm.assemble(ChemClass.SIGE, ["dichlorosilane", "germane", "hcl"], Mode.BLANKET)
    assert net.uses_Ge_model
    assert not net.uses_B_model
    assert net.dopant is None


def test_assemble_sigeb_with_dopant(asm):
    net = asm.assemble(ChemClass.SIGE_B, ["dichlorosilane", "germane", "hcl", "diborane"], Mode.BLANKET)
    assert net.uses_Ge_model
    assert net.uses_B_model


def test_assemble_sigec_with_carbon_source(asm):
    net = asm.assemble(
        ChemClass.SIGEC,
        ["silane", "germane", "methylsilane", "hcl"],
        Mode.BLANKET,
    )
    assert net.uses_Ge_model
    assert net.uses_C_model
    assert net.c_source is not None


def test_assemble_requires_si_source(asm):
    with pytest.raises(ValueError):
        asm.assemble(ChemClass.SIGE, ["germane", "hcl"], Mode.BLANKET)


def test_anti_contamination_b2h6_absent_vs_zero_is_bit_identical():
    """The core correctness guarantee: predicting GR for a SiGe recipe must
    be IDENTICAL whether B2H6 is structurally absent from the feature build
    (ln_B2H6 defaults to 0.0 placeholder, per features.build_features) or
    present with p_B2H6=0. The GR/Ge model never reads the B2H6 column at
    all (gr_logmodel only indexes columns 0-2), so this is true by
    construction -- this test pins that invariant permanently."""
    params = {"lnK_GR": 1.0, "kappa_GR": -2.0, "gamma_HCl": -0.7, "gamma_GeH4": 1.3}

    # Row A: B2H6 "absent" (placeholder 0.0 in the ln_B2H6 column, as
    # features.build_features does when p_B2H6 <= 0).
    X_absent = jnp.array([[0.1, -0.3, -3.0, 0.0]])
    # Row B: same physical recipe, but with an arbitrary large ln_B2H6 value,
    # simulating "present but the model shouldn't care because GR is a pure
    # SiGe recipe and gr_logmodel structurally never reads column 3".
    X_present = jnp.array([[0.1, -0.3, -3.0, 5.0]])

    gr_absent = gr_logmodel(params, X_absent)
    gr_present = gr_logmodel(params, X_present)
    assert jnp.array_equal(gr_absent, gr_present), "GR must not depend on B2H6 column at all"
