# Serving capacity, measured

Every figure here comes from a run on the machine or instance named
beside it. Nothing is extrapolated.

## Where the time actually goes

A single scoring call, broken down on the laptop (i5-11320H, Windows 11,
Python 3.13, scikit-learn 1.9), batch of 10 customers:

| Stage                                   | Time    | Share |
| --------------------------------------- | ------- | ----- |
| Building the request DataFrame           | 2.00 ms | 13%   |
| `ColumnTransformer` preprocessing        | 12.39 ms | 85%  |
| `LogisticRegression.predict_proba`       | 0.19 ms | 1.5%  |

The model is 1.5% of the cost of serving it. Imputation with missingness
indicators, scaling and one-hot encoding are the other 98.5%. Anyone
tuning this service for latency should start with the transformer and
leave the model alone.

## The cost is per call, so batching is the whole story

Same machine, same model, varying the batch size:

| Rows per request | ms per call | ms per row | Rows per second |
| ---------------- | ----------- | ---------- | --------------- |
| 1                | 15.79       | 15.79      | 63              |
| 10               | 13.22       | 1.32       | 756             |
| 50               | 13.11       | 0.26       | 3,813           |
| 100              | 12.05       | 0.12       | 8,296           |
| 500              | 13.21       | 0.026      | 37,864          |
| 1,000            | 13.36       | 0.013      | 74,827          |

The per-call cost is flat at roughly 13 ms from one row to a thousand,
because it is fixed overhead in the transform path rather than work
proportional to the data. Throughput therefore scales almost linearly
with batch size, and the API caps a request at 1,000 customers.

Put in terms of the dataset: the full out-of-time cohort of 64,089
customers scores in under a second at the largest batch size, and takes
about seventeen minutes one row at a time.

## Under concurrent load

Eight concurrent clients, 20 seconds, 100 customers per request, against
the service on the same laptop:

```
requests completed   741        failed   0
rows scored          74,100
requests per second  36.9
rows per second      3,690
latency  p50 205.8 ms   p95 299.2 ms   p99 336.6 ms
```

Throughput per row is lower than the single-threaded table above because
eight threads contend for one interpreter. One uvicorn worker is the
limit being measured here, and more workers is the obvious lever.

Reproduce with:

```bash
olist-retention serve
py scripts/loadtest.py http://127.0.0.1:8000/ --concurrency 8 --seconds 20 --batch-size 100
```

## The load generator has to look like the training data

The first version of `loadtest.py` invented plausible numbers for each
field. The service's own drift monitor reported a maximum population
stability index of **11.57** with six features flagged as material, which
was correct: the traffic genuinely did not resemble the training
population, so the run was measuring the monitor rather than the service.

The generator now samples from the reference distribution stored inside
the model card. The same run afterwards:

| | Max PSI | Material features |
| --- | --- | --- |
| Invented values | 11.57 | `top_category`, `customer_region`, `avg_review_score`, `recency_days`, `tenure_days`, `monetary_total` |
| Drawn from the reference | 0.0166 | none |

That is the monitor working in both directions, which is the only way to
know it works at all: it fires when the input population is wrong and
stays quiet when it is right.

Seven of the 22 numeric features are not monitored, and the model card
names them. Their quantile edges collapse because almost every customer
at the training cutoff has made exactly one purchase, so `frequency`,
`n_reviews`, `n_delivered_orders`, `share_credit_card`,
`late_delivery_rate`, `share_low_reviews` and `has_comment_rate` are very
nearly constant. Watching a constant produces alarms about nothing.

## One bug this found

Sampling from the reference put `Infinity` and `NaN` into request bodies,
because the outermost quantile bins are open at one end and the code
closed them using a neighbouring edge that was itself infinite. The
service answered 500, since the value survived the whole pipeline and
only failed when the response was serialised.

Both sides were fixed. The generator closes open bins using a width
computed from finite edges only, and the API rejects a non-finite feature
with a 422 naming the field, which is the difference between an error a
caller can act on and one they cannot.
