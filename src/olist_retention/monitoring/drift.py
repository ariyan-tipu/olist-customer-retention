"""Feature drift, measured by population stability index.

Why PSI rather than a hypothesis test. The question at serving time is
whether today's traffic still looks like the data the model was fitted
on, and what an operator can act on is a magnitude with an agreed
threshold. A Kolmogorov-Smirnov test answers a different question: given
enough requests it rejects on a difference nobody would care about,
because the p-value tracks sample size as much as it tracks the effect.

The reference distribution is captured during training from the training
snapshot itself, so it travels inside the model artefact. A reference
stored anywhere else can drift apart from the model it claims to
describe, which turns the monitor into a source of false alarms.

Bands follow the convention borrowed from credit scoring, where PSI has
been the standard population-shift measure for decades: below 0.10 the
population is stable, 0.10 to 0.25 deserves a look, and 0.25 upwards is
a material shift.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

STABLE_BELOW = 0.10
MATERIAL_AT = 0.25

# Proportions are floored before the logarithm so an empty bin cannot
# produce an infinite index. The floor is small enough to leave a genuine
# empty bin looking severe and finite enough to print.
PROPORTION_FLOOR = 1e-6

OTHER_CATEGORY = "__other__"


@dataclass(frozen=True)
class NumericReference:
    """Quantile bin edges and the training share falling in each."""

    edges: list[float]
    proportions: list[float]
    missing_rate: float


@dataclass(frozen=True)
class CategoricalReference:
    """Training share per category, with a bucket for the long tail."""

    categories: list[str]
    proportions: list[float]
    missing_rate: float


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    band: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def band_for(psi: float) -> str:
    if psi < STABLE_BELOW:
        return "stable"
    if psi < MATERIAL_AT:
        return "moderate"
    return "material"


def _floor(proportions: np.ndarray) -> np.ndarray:
    return np.clip(proportions, PROPORTION_FLOOR, None)


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population stability index between two proportion vectors."""
    expected = _floor(np.asarray(expected, dtype=float))
    actual = _floor(np.asarray(actual, dtype=float))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def build_numeric_reference(values: np.ndarray, bins: int = 10) -> NumericReference | None:
    """Quantile edges from the training column, or None if degenerate.

    Features with heavy ties (a share that is zero for most customers,
    for instance) collapse to fewer distinct edges than requested, and
    duplicate edges would create empty bins that report drift when
    nothing moved. Such a column is dropped from the monitor and said so
    in the note, since a silent drop is worse than a missing signal.
    """
    values = np.asarray(values, dtype=float)
    missing_rate = float(np.mean(np.isnan(values))) if values.size else 1.0
    present = values[~np.isnan(values)]
    if present.size == 0:
        return None

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.nanquantile(present, quantiles))
    if edges.size < 3:
        return None

    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(present, bins=edges)
    proportions = counts / float(present.size)
    return NumericReference(
        edges=[float(edge) for edge in edges],
        proportions=[float(value) for value in proportions],
        missing_rate=missing_rate,
    )


def build_categorical_reference(
    values: list[Any], max_categories: int = 20
) -> CategoricalReference:
    """Category shares, with everything past the head folded into one bucket."""
    cleaned = [OTHER_CATEGORY if value is None else str(value) for value in values]
    total = float(len(cleaned)) or 1.0
    missing_rate = cleaned.count(OTHER_CATEGORY) / total

    counts: dict[str, int] = {}
    for value in cleaned:
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    head = ranked[:max_categories]
    tail_count = sum(count for _, count in ranked[max_categories:])

    categories = [name for name, _ in head]
    proportions = [count / total for _, count in head]
    if tail_count or OTHER_CATEGORY not in categories:
        categories.append(OTHER_CATEGORY)
        proportions.append(tail_count / total)
    return CategoricalReference(
        categories=categories,
        proportions=proportions,
        missing_rate=missing_rate,
    )


def build_reference(
    frame: Any,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    bins: int = 10,
) -> dict[str, Any]:
    """Serialisable reference distribution for the model card."""
    numeric: dict[str, Any] = {}
    skipped: list[str] = []
    for name in numeric_features:
        reference = build_numeric_reference(frame[name].to_numpy(dtype=float), bins=bins)
        if reference is None:
            skipped.append(name)
            continue
        numeric[name] = asdict(reference)

    categorical = {
        name: asdict(build_categorical_reference(list(frame[name])))
        for name in categorical_features
    }
    return {
        "bins": bins,
        "n_rows": int(len(frame)),
        "numeric": numeric,
        "categorical": categorical,
        "skipped_numeric": skipped,
    }


class DriftMonitor:
    """Rolling window of scored rows, compared against the reference.

    The window is bounded so a long-running process cannot grow without
    limit, and the index is computed on demand rather than per request,
    because scraping happens far less often than scoring does.
    """

    def __init__(self, reference: dict[str, Any] | None, window: int = 2000) -> None:
        self.reference = reference or {}
        self.window = window
        self._rows: deque[dict[str, Any]] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._seen = 0

    @property
    def enabled(self) -> bool:
        return bool(self.reference.get("numeric") or self.reference.get("categorical"))

    def observe(self, rows: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._rows.extend(rows)
            self._seen += len(rows)

    def _numeric_drift(self, name: str, spec: dict[str, Any], column: list[Any]) -> FeatureDrift:
        edges = np.asarray(spec["edges"], dtype=float)
        values = np.asarray(
            [np.nan if value is None else float(value) for value in column], dtype=float
        )
        present = values[~np.isnan(values)]
        if present.size == 0:
            return FeatureDrift(name, 0.0, "stable", "no non-missing values in the window")
        counts, _ = np.histogram(present, bins=edges)
        actual = counts / float(present.size)
        return FeatureDrift(name, round(_psi(spec["proportions"], actual), 6), "")

    def _categorical_drift(
        self, name: str, spec: dict[str, Any], column: list[Any]
    ) -> FeatureDrift:
        categories: list[str] = spec["categories"]
        index = {category: position for position, category in enumerate(categories)}
        counts = np.zeros(len(categories), dtype=float)
        other = index.get(OTHER_CATEGORY, len(categories) - 1)
        for value in column:
            key = OTHER_CATEGORY if value is None else str(value)
            counts[index.get(key, other)] += 1.0
        total = counts.sum() or 1.0
        return FeatureDrift(name, round(_psi(spec["proportions"], counts / total), 6), "")

    def snapshot(self) -> dict[str, Any]:
        """Per-feature drift over the current window."""
        with self._lock:
            rows = list(self._rows)
            seen = self._seen

        if not self.enabled:
            return {"enabled": False, "reason": "model artefact carries no reference distribution"}
        if len(rows) < 30:
            return {
                "enabled": True,
                "window_rows": len(rows),
                "rows_scored_total": seen,
                "status": "warming up",
                "features": [],
            }

        results: list[FeatureDrift] = []
        for name, spec in self.reference.get("numeric", {}).items():
            results.append(self._numeric_drift(name, spec, [row.get(name) for row in rows]))
        for name, spec in self.reference.get("categorical", {}).items():
            results.append(self._categorical_drift(name, spec, [row.get(name) for row in rows]))

        for result in results:
            if not result.band:
                result.band = band_for(result.psi)
        results.sort(key=lambda item: item.psi, reverse=True)

        worst = results[0] if results else None
        return {
            "enabled": True,
            "window_rows": len(rows),
            "rows_scored_total": seen,
            "status": "ok",
            "max_psi": worst.psi if worst else 0.0,
            "max_psi_feature": worst.feature if worst else "",
            "material_features": [item.feature for item in results if item.band == "material"],
            "features": [item.as_dict() for item in results],
        }
