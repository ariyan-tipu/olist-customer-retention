"""The S3 artefact store, against a mocked AWS API.

moto stands in for S3 so this path is exercised on every push without an
account, a card or a network call. The behaviour worth pinning is the
refusal: the artefact is a pickle, so anything that arrives without a
matching digest has to stop before it reaches joblib.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from olist_retention.artefacts.s3 import (
    MANIFEST_SUFFIX,
    ArtefactError,
    ArtefactStore,
    parse_uri,
    sha256_of,
)

BUCKET = "olist-artefacts-test"
REGION = "eu-west-2"


@pytest.fixture
def bucket():
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION}
        )
        yield client


@pytest.fixture
def artefact(tmp_path):
    path = tmp_path / "churn_model.joblib"
    path.write_bytes(b"pretend this is a joblib pickle")
    (tmp_path / "model_card.json").write_text(
        json.dumps({"model_name": "logistic_regression", "artefact_version": 2}),
        encoding="utf-8",
    )
    return path


def test_publish_then_fetch_round_trip(bucket, artefact, tmp_path):
    store = ArtefactStore(bucket=BUCKET, client=bucket)
    published = store.publish(artefact)

    assert published.uri == f"s3://{BUCKET}/models/churn_model.joblib"
    assert published.model_name == "logistic_regression"
    assert published.sha256 == sha256_of(artefact)

    destination = tmp_path / "downloaded" / "churn_model.joblib"
    store.fetch(published.uri, destination)
    assert destination.read_bytes() == artefact.read_bytes()


def test_the_manifest_records_what_was_published(bucket, artefact):
    store = ArtefactStore(bucket=BUCKET, client=bucket)
    published = store.publish(artefact)

    body = bucket.get_object(Bucket=BUCKET, Key="models/churn_model.joblib" + MANIFEST_SUFFIX)
    manifest = json.loads(body["Body"].read())
    assert manifest["sha256"] == published.sha256
    assert manifest["model_name"] == "logistic_regression"
    assert manifest["artefact_version"] == 2


def test_a_tampered_object_is_refused(bucket, artefact, tmp_path):
    store = ArtefactStore(bucket=BUCKET, client=bucket)
    published = store.publish(artefact)

    # Replace the object, leaving the manifest describing the old one.
    bucket.put_object(
        Bucket=BUCKET, Key="models/churn_model.joblib", Body=b"a different payload entirely"
    )

    with pytest.raises(ArtefactError, match="Digest mismatch"):
        store.fetch(published.uri, tmp_path / "out.joblib")
    assert not (tmp_path / "out.joblib").exists()


def test_an_object_without_a_manifest_is_refused(bucket, tmp_path):
    bucket.put_object(Bucket=BUCKET, Key="models/loose.joblib", Body=b"no manifest here")
    store = ArtefactStore(bucket=BUCKET, client=bucket)

    with pytest.raises(ArtefactError, match="No manifest"):
        store.fetch(f"s3://{BUCKET}/models/loose.joblib", tmp_path / "out.joblib")


def test_publishing_a_missing_file_says_so(bucket, tmp_path):
    store = ArtefactStore(bucket=BUCKET, client=bucket)
    with pytest.raises(ArtefactError, match="No artefact to publish"):
        store.publish(tmp_path / "absent.joblib")


@pytest.mark.parametrize(
    "uri",
    ["https://example.com/model", "s3://", "s3://bucket-only", "not a uri at all"],
)
def test_bad_uris_are_rejected(uri):
    with pytest.raises(ArtefactError):
        parse_uri(uri)


def test_a_good_uri_splits_into_bucket_and_key():
    assert parse_uri("s3://a-bucket/nested/key.joblib") == ("a-bucket", "nested/key.joblib")
