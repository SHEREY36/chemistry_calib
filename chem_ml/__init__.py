"""
chem_ml: Physics-ML Chemistry Calibration framework (Tomasini et al. 2010
reproduction, Phases 0-8). See build_steps_and_cfd_integration.md for the
full phase spec.

INVARIANT (float64 everywhere): kinetics fits are ill-conditioned in
float32 -- enable x64 globally before any other JAX usage in the package.
"""
import os

import jax
import numpyro

jax.config.update("jax_enable_x64", True)
# Let NUTS run chains in parallel on CPU cores instead of sequentially
# (must be set before any JAX op locks in the device count).
numpyro.set_host_device_count(min(4, os.cpu_count() or 1))
