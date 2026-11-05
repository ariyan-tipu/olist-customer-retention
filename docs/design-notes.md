# Design notes

The decisions behind the pipeline, in full. The README keeps the short
version.

## The selection rule, and what evidence there is for it

"We chose the metric first" is the kind of claim that is easy to make
afterwards. The rule lives in
[`models/train.py`](../src/olist_retention/models/train.py):
cross-validation runs on the training snapshot only, the winner is chosen on
its mean average precision, and the out-of-time snapshot is loaded afterwards
purely to score the model already picked. That structure is what enforces the
order, and it is worth being straight that it is the only evidence here. There
is no timestamped record proving the metric was fixed before the holdout was
ever examined, so read it as a design that makes the mistake hard to make,
rather than an audited pre-registration.

## The identity trap

Olist issues a fresh `customer_id` per order; the person is
`customer_unique_id`. Joining on the wrong one silently turns customer
analysis into order analysis. The grain is asserted in schema docs, enforced
in validation and pinned by tests.

## Point-in-time discipline

Features at a cutoff may only use what was knowable then: purchases count once
placed, but delivery features require the parcel to have arrived by the cutoff
and review features require the review to have been written. A handcrafted
seven-customer dataset pins each rule, including a customer whose order was in
transit across the cutoff.

The same tests caught a real bug: naive datetime literals crossed a British
daylight-saving change and shifted every boundary by an hour, which a UTC-only
CI machine would never have seen. Timestamp literals are now rendered as
strings and parsed in the session's pinned UTC timezone.

## Spark decisions, made for reasons

The products and category dimensions are explicitly broadcast, turning the
basket join into a map-side hash join with no shuffle of the item rows.
Shuffle partitions are set to 16, not the 200 default, because at 100k orders
the default creates hundreds of near-empty tasks whose scheduling overhead
dominates. The reused purchase spine is cached once, deliberately, and
everything else stays lazy so Spark can collapse the plan. Snapshots are
written as single parquet files because partitioning below a few million rows
adds bookkeeping and buys no speed.

## Evaluation under a 1% base rate

With a 0.975% base rate, ROC-AUC flatters: the gradient booster "wins" ROC
while losing average precision, which is the metric aligned with actually
targeting anyone. Selection runs on cross-validated average precision over the
training snapshot, and the holdout is only loaded afterwards to score the
winner.

Class weighting makes the scores rank well but destroys calibration, so the
service's outputs are treated as rankings; recalibration on a held-out slice is
the stated next step before anyone reads them as probabilities. Class weighting
pushed predicted probabilities into the 0.25 to 0.65 range while the observed
repeat rate never leaves about 2%, so the numbers order customers correctly and
mean nothing as probabilities. A model that is useful and badly calibrated at
the same time is normal; quoting its output as "a 60% chance of repurchase"
would not be.

The out-of-time evaluation also surfaces genuine drift: the base rate moves
from 1.286% to 0.975% between cutoffs, which is production reality rather than
an inconvenience. Permutation importance is measured on the holdout rather than
in-sample, so it reports what actually carried signal out of time.

## Censoring is not churn

98.1% of survival spells are censored. The
[survival write-up](survival-analysis.md) spells out why deleting or
mislabelling those customers wrecks every estimate, and the CI stats job proves
the machinery end to end: the synthetic generator plants known effects (late
delivery and bad reviews stretch the time to the next purchase), R must recover
both directions significantly, and Octave must reproduce R's fit from the raw
likelihood, on every push.

## SQL that is tested

The Databricks SQL views in `sql/` execute against Spark temporary views inside
the test suite, so a broken view fails the build rather than a dashboard.
