"""Power analysis for a two-arm experiment on a binary outcome.

The scenario: operations can pay for faster carriers in a treatment group,
and the survival analysis says late deliveries slow repeat purchase. Before
running that experiment you need to know how many customers it takes to
detect a plausible uplift in the repeat rate, and what error rates you are
signing up for.

Everything here is implemented from the definitions rather than imported,
because reasoning about Type I and Type II error is the point:

* Type I error (alpha): the test says the treatment worked when it did
  nothing. Fixed by the significance level.
* Type II error (beta): the treatment worked but the test missed it.
  Power is 1 - beta and is bought with sample size.

The closed-form sizes use the standard normal approximation with pooled
variance under the null. A Monte Carlo simulation then checks the algebra
by measuring both error rates empirically; the unit tests additionally
cross-check against statsmodels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import ceil, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from olist_retention.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SimulationResult:
    replications: int
    empirical_power: float
    empirical_alpha: float


@dataclass(frozen=True)
class PowerScenario:
    baseline_rate: float
    relative_uplift: float
    treated_rate: float
    alpha: float
    target_power: float
    n_per_arm: int
    analytic_power_at_n: float
    simulation: SimulationResult


class TwoProportionPower:
    """Design calculations for a two-proportion z-test."""

    def __init__(self, alpha: float, target_power: float, two_sided: bool = True) -> None:
        if not 0 < alpha < 1 or not 0 < target_power < 1:
            raise ValueError("alpha and target_power must be in (0, 1)")
        self.alpha = alpha
        self.target_power = target_power
        self.two_sided = two_sided

    @property
    def _z_alpha(self) -> float:
        tail = self.alpha / 2 if self.two_sided else self.alpha
        return float(norm.ppf(1 - tail))

    def required_n_per_arm(self, p0: float, p1: float) -> int:
        """Smallest n per arm reaching the target power for p0 versus p1."""
        _validate_rates(p0, p1)
        z_a = self._z_alpha
        z_b = float(norm.ppf(self.target_power))
        p_bar = (p0 + p1) / 2
        numerator = (
            z_a * sqrt(2 * p_bar * (1 - p_bar)) + z_b * sqrt(p0 * (1 - p0) + p1 * (1 - p1))
        ) ** 2
        return ceil(numerator / (p1 - p0) ** 2)

    def power_at(self, n_per_arm: int, p0: float, p1: float) -> float:
        """Analytic power of the test at a given sample size."""
        _validate_rates(p0, p1)
        if n_per_arm < 1:
            raise ValueError("n_per_arm must be positive")
        z_a = self._z_alpha
        p_bar = (p0 + p1) / 2
        effect = abs(p1 - p0) * sqrt(n_per_arm)
        null_sd = sqrt(2 * p_bar * (1 - p_bar))
        alt_sd = sqrt(p0 * (1 - p0) + p1 * (1 - p1))
        return float(norm.cdf((effect - z_a * null_sd) / alt_sd))

    def simulate(
        self, p0: float, p1: float, n_per_arm: int, replications: int, seed: int
    ) -> SimulationResult:
        """Measure empirical power and empirical alpha by simulation.

        Power is estimated under the alternative (p1 in the treatment
        arm); alpha is estimated by rerunning the same test with the
        treatment effect switched off, which should reject at close to
        the nominal rate if the approximation is sound.
        """
        rng = np.random.default_rng(seed)
        power = self._rejection_rate(rng, p0, p1, n_per_arm, replications)
        alpha = self._rejection_rate(rng, p0, p0, n_per_arm, replications)
        return SimulationResult(
            replications=replications,
            empirical_power=round(power, 4),
            empirical_alpha=round(alpha, 4),
        )

    def _rejection_rate(
        self, rng: np.random.Generator, p0: float, p1: float, n: int, reps: int
    ) -> float:
        control = rng.binomial(n, p0, size=reps)
        treatment = rng.binomial(n, p1, size=reps)
        pooled = (control + treatment) / (2 * n)
        se = np.sqrt(np.clip(pooled * (1 - pooled) * 2 / n, 1e-12, None))
        z = (treatment - control) / n / se
        rejected = np.abs(z) > self._z_alpha if self.two_sided else z > self._z_alpha
        return float(np.mean(rejected))

    # ------------------------------------------------------------------

    def run_grid(
        self,
        baseline_rate: float,
        relative_uplifts: tuple[float, ...],
        replications: int,
        seed: int,
    ) -> list[PowerScenario]:
        scenarios = []
        for uplift in relative_uplifts:
            p1 = baseline_rate * (1 + uplift)
            n = self.required_n_per_arm(baseline_rate, p1)
            simulation = self.simulate(baseline_rate, p1, n, replications, seed)
            scenarios.append(
                PowerScenario(
                    baseline_rate=round(baseline_rate, 6),
                    relative_uplift=uplift,
                    treated_rate=round(p1, 6),
                    alpha=self.alpha,
                    target_power=self.target_power,
                    n_per_arm=n,
                    analytic_power_at_n=round(self.power_at(n, baseline_rate, p1), 4),
                    simulation=simulation,
                )
            )
            logger.info(
                "Uplift %+.0f%%: %.3f%% -> %.3f%% needs %d per arm "
                "(simulated power %.3f, simulated alpha %.3f)",
                100 * uplift,
                100 * baseline_rate,
                100 * p1,
                n,
                simulation.empirical_power,
                simulation.empirical_alpha,
            )
        return scenarios

    def write_report(self, scenarios: list[PowerScenario], out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "power_analysis.json"
        report_path.write_text(
            json.dumps([asdict(s) for s in scenarios], indent=2) + "\n", encoding="utf-8"
        )
        plot_path = out_dir / "power_curves.png"
        self._plot(scenarios, plot_path)
        logger.info("Wrote power analysis to %s", report_path)
        return report_path, plot_path

    def _plot(self, scenarios: list[PowerScenario], path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for scenario in scenarios:
            n_star = scenario.n_per_arm
            grid = np.unique(np.linspace(max(200, n_star // 4), n_star * 2, 60).astype(int))
            powers = [
                self.power_at(int(n), scenario.baseline_rate, scenario.treated_rate) for n in grid
            ]
            ax.plot(
                grid,
                powers,
                label=f"+{scenario.relative_uplift:.0%} uplift (n*={n_star:,})",
            )
        ax.axhline(
            self.target_power, linestyle="--", color="grey", linewidth=1, label="target power"
        )
        ax.set_xlabel("Customers per arm")
        ax.set_ylabel("Power")
        ax.set_title("Power against sample size for the delivery experiment")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def _validate_rates(p0: float, p1: float) -> None:
    for rate in (p0, p1):
        if not 0 < rate < 1:
            raise ValueError("rates must be strictly between 0 and 1")
    if p0 == p1:
        raise ValueError("p0 and p1 must differ to size an experiment")
