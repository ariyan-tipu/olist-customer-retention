"""CI gate: the Cox fit must recover the synthetic generator's ground truth.

The generator stretches time to the next purchase after a late delivery
and after a bad review, so the fitted hazard ratios for both must be
below one and clearly significant. If a refactor breaks a join, a filter
or the censoring logic, the effects wash out and this gate fails, which
is a far stronger signal than row counts matching.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HAZARD_RATIOS = Path("outputs") / "survival" / "cox_hazard_ratios.csv"
SIGNIFICANCE = 0.05


def main() -> int:
    with HAZARD_RATIOS.open(encoding="utf-8") as handle:
        rows = {row["covariate"]: row for row in csv.DictReader(handle)}

    checks = [
        ("late_delivery", "late delivery slows repurchase"),
        ("review_grouplow", "a bad review slows repurchase"),
    ]
    failures: list[str] = []
    for covariate, meaning in checks:
        if covariate not in rows:
            failures.append(f"{covariate}: missing from the Cox output")
            continue
        hazard_ratio = float(rows[covariate]["hazard_ratio"])
        p_value = float(rows[covariate]["p_value"])
        status = "ok"
        if hazard_ratio >= 1.0:
            status = "WRONG DIRECTION"
            failures.append(f"{covariate}: hazard ratio {hazard_ratio} is not below 1")
        elif p_value >= SIGNIFICANCE:
            status = "NOT SIGNIFICANT"
            failures.append(f"{covariate}: p value {p_value} is not below {SIGNIFICANCE}")
        print(f"{covariate:20s} HR={hazard_ratio:6.3f} p={p_value:<10.3g} {status} ({meaning})")

    if failures:
        print("\nPlanted effect recovery FAILED:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("\nPlanted effects recovered correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
