# Running the scorer on AWS

The model in this repository is trained on a laptop and served from an
EC2 instance that reads it out of S3. This document is the whole
procedure, and the two scripts under `deploy/aws/` are the whole
implementation.

## What gets created

| Resource | Purpose | Cost posture |
| --- | --- | --- |
| S3 bucket, private and versioned | holds `churn_model.joblib` and its manifest | pennies at this size |
| IAM role plus instance profile | lets the instance read that one prefix and publish metrics in one namespace | free |
| Security group | port 80 to the world, port 22 to a single address | free |
| EC2 `t3.micro` with a 12 GiB gp3 volume | runs uvicorn behind nginx | the only meaningful line |
| CloudWatch custom metrics | seven gauges, published once a minute | inside the always-free allowance |

There is no load balancer, no NAT gateway and no managed database. Each
of those is a fixed monthly charge for something this workload does not
need, and the last two are the usual way a free-tier account starts
costing money by accident.

## The cost guarantee

The account is opened on the AWS **Free plan**, where AWS states that no
charges are incurred and that it will not charge the payment method
until the account is upgraded. When the credits are used or six months
pass, AWS closes the account rather than billing. That closure is the
reason the repository carries captured evidence of the deployment
running: the measurements outlive the instance.

Consumption is roughly 0.0114 USD an hour for the instance, so a full
six months lands near 55 USD against 100 to 200 USD of credits.

## Before you start

* An AWS account on the Free plan, with a zero-spend budget configured.
* A local profile in `~/.aws/credentials` holding a bootstrap key.
* A trained model at `models/churn_model.joblib`, which means
  `olist-retention train` has been run at least once.

## Deploy

```bash
# 1. Put the model somewhere the instance can read it.
olist-retention publish --bucket <your-bucket> --profile olist

# 2. Create everything else and launch the instance.
py deploy/aws/bootstrap.py --profile olist --bucket <your-bucket>
```

The bootstrap prints the public address when the instance is running.
First boot installs the package from GitHub and pulls the model, which
takes a few minutes; `http://<address>/readyz` returns 200 when that has
finished.

```bash
curl -s http://<address>/readyz
curl -s -X POST http://<address>/v1/score \
  -H "Content-Type: application/json" \
  -d '{"customers": [{"recency_days": 30, "frequency": 2, "avg_review_score": 5}]}'
curl -s http://<address>/v1/monitoring
curl -s http://<address>/metrics
```

Both scripts are safe to run again. The bootstrap checks for each
resource before creating it, so a half-finished run is fixed by
repeating it.

## Measure it

```bash
py scripts/loadtest.py http://<address>/ --concurrency 16 --seconds 60 \
  --json-out results/loadtest.json
```

This reports requests a second and the latency spread that came with it.
The tail matters more than the middle here: a demonstration shows the
median, and an operator is woken by the 99th percentile.

## What the instance can and cannot do

The security posture is the part worth explaining out loud:

* No AWS key exists on the instance. It authenticates through an
  instance profile, so there is no credential on the box to steal.
* Its policy grants `s3:GetObject` on the `models/` prefix of one
  bucket, `s3:ListBucket` limited to the same prefix, and
  `cloudwatch:PutMetricData` limited to one namespace. It cannot write
  to the bucket it reads from, which means a compromise of the service
  cannot replace the model it serves.
* The bucket blocks public access at the account boundary, is versioned
  so an overwritten model can be recovered, and is encrypted at rest.
* Port 22 is open to one address, defaulting to the address of the
  machine that ran the bootstrap. Port 80 is deliberately public,
  because a demonstration nobody can reach demonstrates nothing.
* The systemd unit runs as an unprivileged user under `ProtectSystem`,
  `PrivateTmp` and `NoNewPrivileges`, with one writable directory.

## Monitoring

Once traffic arrives, `/v1/monitoring` reports latency percentiles and a
population stability index per feature, and `/metrics` reports the same
figures for a Prometheus scraper. Seven of those numbers also reach
CloudWatch every minute, which is where an alarm would hang if this were
carrying real customers. The count is capped at seven so it stays inside
the ten custom metrics that are free indefinitely.

Drift bands follow the credit-scoring convention: below 0.10 stable,
0.10 to 0.25 worth a look, 0.25 and above a material shift in who is
being scored. A material reading is a prompt to retrain, and the number
that triggers it names the feature that moved.

## What this is not

A single instance in one availability zone, serving plain HTTP, with no
autoscaling and no failover. It is sized for a demonstration and for
capacity measurement.

Making it production-grade for real customers would mean TLS with a
domain in front, at least two instances behind a load balancer, an
alarm on the drift metric wired to a retraining job, and the model
artefact promoted through an approval step rather than published from a
laptop. Each of those is an addition to this design instead of a rewrite
of it, and each also costs money, which is why the free-tier version
stops here.

## Teardown

```bash
py deploy/aws/teardown.py --profile olist                    # keeps the bucket
py deploy/aws/teardown.py --profile olist --delete-bucket    # removes everything
```

Then delete the `bootstrap-admin` IAM user's access key, and confirm in
Billing and Cost Management that nothing is running.
