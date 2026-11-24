"""Pair every historical correlation estimate with its corrected-mean counterpart."""

from pathlib import Path
import argparse
import pandas as pd


def run(root):
    keys = ["specification", "exposure", "outcome"]
    before = pd.read_csv(root / "results/all_associations.csv")
    after = pd.read_csv(root / "corrected_results/all_associations.csv")
    estimates = before.merge(
        after, on=keys, suffixes=("_original", "_corrected"), validate="one_to_one"
    )
    assert len(estimates) == len(before) == len(after) == 768
    assert estimates.n_original.equals(estimates.n_corrected)
    estimates["delta_r"] = estimates.r_corrected - estimates.r_original
    estimates["BH_threshold_changed"] = (estimates.q_BH_original < 0.05) != (
        estimates.q_BH_corrected < 0.05
    )
    estimates.to_csv(root / "outcome_audit/correlation_versions.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    run(p.parse_args().root)
