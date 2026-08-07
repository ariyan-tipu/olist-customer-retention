"""Measure what the deployed scorer can actually take.

    py scripts/loadtest.py http://<address>/ --concurrency 16 --seconds 30

Standard library only, so it runs anywhere the repository does without
adding a dependency for something used a handful of times.

Why this exists. A dataset of ninety-nine thousand orders says nothing
about how a service behaves under traffic, and "it works on my laptop"
is not a capacity claim. This turns the question into a measurement:
requests a second at a stated batch size, with the latency spread that
came with it. The number it produces is small, and a small measured
number is worth more than a large estimated one.

Latency is reported at the tail as well as the middle, because the
middle is what a demonstration shows and the tail is what an operator
gets paged about.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

DEFAULT_PROFILE = Path("models/model_card.json")
FALLBACK_REGIONS = ("sudeste", "nordeste", "sul", "norte", "centro_oeste")


class TrafficProfile:
    """Draws request rows that look like the data the model was fitted on.

    The first version of this script invented plausible-looking numbers,
    and the deployed service's own drift monitor immediately reported six
    features as materially shifted. The load test was measuring the
    monitor rather than the service. Sampling from the reference
    distribution inside the model card fixes that at the source: a
    capacity run now leaves the drift index where it started, and a
    reading above the band during one means something real.

    Features the reference skipped as near-constant are left out of the
    payload, which is what the monitor does with them too.
    """

    def __init__(self, reference: dict | None) -> None:
        self.reference = reference or {}
        self.numeric = self.reference.get("numeric", {})
        self.categorical = self.reference.get("categorical", {})

    @classmethod
    def load(cls, path: Path) -> TrafficProfile:
        if not path.is_file():
            return cls(None)
        card = json.loads(path.read_text(encoding="utf-8"))
        return cls(card.get("reference_distribution"))

    @property
    def informed(self) -> bool:
        return bool(self.numeric or self.categorical)

    @staticmethod
    def _pick(rng: random.Random, weights: list[float]) -> int:
        total = sum(weights) or 1.0
        target = rng.random() * total
        running = 0.0
        for index, weight in enumerate(weights):
            running += weight
            if target <= running:
                return index
        return len(weights) - 1

    def _numeric_value(self, rng: random.Random, spec: dict) -> float | None:
        if rng.random() < spec.get("missing_rate", 0.0):
            return None
        edges = list(spec["edges"])
        finite = [edge for edge in edges if math.isfinite(edge)]
        if not finite:
            return None

        index = self._pick(rng, list(spec["proportions"]))
        low, high = edges[index], edges[index + 1]

        # The outermost bins are open at one end. They are closed here
        # using one typical bin width, computed from the finite edges
        # only. Reading the width off a neighbouring edge is what an
        # earlier version did, and on a feature with few distinct
        # quantiles that neighbour is itself infinite, which put NaN and
        # Infinity into the payload and gave the service a 500.
        span = max(finite) - min(finite)
        typical = (span / max(len(finite) - 1, 1)) or 1.0
        if not math.isfinite(low):
            low = high - typical
        if not math.isfinite(high):
            high = low + typical

        value = round(rng.uniform(low, high), 4)
        return value if math.isfinite(value) else None

    def row(self, rng: random.Random) -> dict:
        if not self.informed:
            return {
                "recency_days": round(rng.gauss(120, 60), 2),
                "frequency": rng.randint(1, 4),
                "avg_review_score": round(rng.uniform(1, 5), 1),
                "customer_region": rng.choice(FALLBACK_REGIONS),
            }
        row: dict = {}
        for name, spec in self.numeric.items():
            value = self._numeric_value(rng, spec)
            if value is not None:
                row[name] = value
        for name, spec in self.categorical.items():
            index = self._pick(rng, list(spec["proportions"]))
            category = spec["categories"][index]
            if not category.startswith("__"):
                row[name] = category
        return row


@dataclass
class Results:
    latencies_ms: list[float] = field(default_factory=list)
    failures: int = 0
    rows: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, duration_ms: float, rows: int) -> None:
        with self.lock:
            self.latencies_ms.append(duration_ms)
            self.rows += rows

    def fail(self) -> None:
        with self.lock:
            self.failures += 1


def worker(
    url: str,
    rng_seed: int,
    batch_size: int,
    deadline: float,
    results: Results,
    timeout: float,
    profile: TrafficProfile,
) -> None:
    rng = random.Random(rng_seed)
    while time.monotonic() < deadline:
        batch = {"customers": [profile.row(rng) for _ in range(batch_size)]}
        payload = json.dumps(batch).encode("utf-8")
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response.read()
        except (urllib.error.URLError, TimeoutError, OSError):
            results.fail()
            continue
        results.record((time.perf_counter() - started) * 1000.0, batch_size)


def percentile(ordered: list[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url", help="Service root, for example http://1.2.3.4/")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--json-out", default=None, help="Write the summary to this path")
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="Model card to draw request rows from (default: models/model_card.json)",
    )
    args = parser.parse_args(argv)

    profile = TrafficProfile.load(args.profile)
    if profile.informed:
        print(f"Drawing traffic from the reference distribution in {args.profile}")
    else:
        print(
            f"No reference distribution at {args.profile}. Falling back to invented "
            "values, which will register as feature drift on the service.",
            file=sys.stderr,
        )

    url = urljoin(args.base_url if args.base_url.endswith("/") else args.base_url + "/", "v1/score")
    print(f"Warming up against {url}")
    try:
        worker(url, 0, args.batch_size, time.monotonic() + 2.0, Results(), args.timeout, profile)
    except Exception as error:  # noqa: BLE001 - a clear message beats a stack trace
        print(f"Could not reach the service: {error}", file=sys.stderr)
        return 1

    results = Results()
    deadline = time.monotonic() + args.seconds
    started = time.perf_counter()
    threads = [
        threading.Thread(
            target=worker,
            args=(url, seed, args.batch_size, deadline, results, args.timeout, profile),
            daemon=True,
        )
        for seed in range(args.concurrency)
    ]
    print(
        f"Running {args.concurrency} clients for {args.seconds}s "
        f"at {args.batch_size} customers per request"
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started

    ordered = sorted(results.latencies_ms)
    completed = len(ordered)
    summary = {
        "url": url,
        "traffic_drawn_from_reference": profile.informed,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "duration_seconds": round(elapsed, 2),
        "requests_completed": completed,
        "requests_failed": results.failures,
        "rows_scored": results.rows,
        "requests_per_second": round(completed / elapsed, 1) if elapsed else 0.0,
        "rows_per_second": round(results.rows / elapsed, 1) if elapsed else 0.0,
        "latency_ms": {
            "mean": round(statistics.fmean(ordered), 2) if ordered else 0.0,
            "p50": round(percentile(ordered, 0.50), 2),
            "p95": round(percentile(ordered, 0.95), 2),
            "p99": round(percentile(ordered, 0.99), 2),
            "max": round(ordered[-1], 2) if ordered else 0.0,
        },
    }

    print()
    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        print(f"\nWritten to {args.json_out}")
    return 0 if results.failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
