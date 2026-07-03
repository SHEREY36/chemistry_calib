"""
Entry points: calibrate, predict, inverse, sensitivity, add-reactor,
add-species, export-mechanism, active-learn (Phase 11). Filled in as the
corresponding phase lands; `calibrate` is wired first since it's the
Phase 4 reproduction gate.
"""
from __future__ import annotations

import argparse
import logging

from chem_ml.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(prog="chem-ml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate", help="Run Phase 1-4 ingest + calibration against Tomasini DS1/DS2")
    args = parser.parse_args()

    cfg = Config()
    if args.command == "calibrate":
        from chem_ml.pipeline import load_all_datasets
        load_all_datasets(cfg)


if __name__ == "__main__":
    main()
