"""Delete everything bootstrap.py created, so the account returns to empty.

    py deploy/aws/teardown.py --profile olist

Reads the state file the bootstrap wrote. Ordering matters: the instance
has to go before the security group it sits in, and the role has to come
out of the instance profile before either can be deleted. Anything
already gone is treated as success, so a partial teardown can be
finished by running this again.

The bucket is kept unless --delete-bucket is passed. Losing a trained
model to a stray teardown is a worse outcome than a few kilobytes of
storage sitting in an account.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

STATE_PATH = Path(__file__).with_name(".stack-state.json")


def log(message: str) -> None:
    print(f"[teardown] {message}", flush=True)


def ignore_missing(action: Any, *args: Any, **kwargs: Any) -> None:
    try:
        action(*args, **kwargs)
    except ClientError as error:
        code = error.response["Error"]["Code"]
        if code in {
            "NoSuchEntity",
            "InvalidInstanceID.NotFound",
            "InvalidGroup.NotFound",
            "InvalidKeyPair.NotFound",
            "NoSuchBucket",
        }:
            log(f"already gone ({code})")
            return
        raise


def empty_bucket(session: boto3.Session, bucket: str) -> None:
    """Remove every object and version so the bucket can be deleted."""
    resource = session.resource("s3")
    target = resource.Bucket(bucket)
    target.object_versions.delete()
    log(f"emptied {bucket}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument(
        "--delete-bucket",
        action="store_true",
        help="Also delete the model bucket and every version in it",
    )
    args = parser.parse_args(argv)

    if not args.state.is_file():
        raise SystemExit(f"No state file at {args.state}; nothing recorded to delete")
    state = json.loads(args.state.read_text(encoding="utf-8"))

    session = boto3.Session(
        profile_name=args.profile, region_name=args.region or state.get("region")
    )
    ec2 = session.client("ec2")
    iam = session.client("iam")

    instance_id = state.get("instance_id")
    if instance_id:
        log(f"terminating {instance_id}")
        ignore_missing(ec2.terminate_instances, InstanceIds=[instance_id])
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
        log("instance terminated, and its root volume with it")

    role = state.get("role_name")
    profile = state.get("instance_profile")
    if role and profile:
        ignore_missing(
            iam.remove_role_from_instance_profile, InstanceProfileName=profile, RoleName=role
        )
        ignore_missing(iam.delete_instance_profile, InstanceProfileName=profile)
        ignore_missing(
            iam.delete_role_policy, RoleName=role, PolicyName="olist-scoring-least-privilege"
        )
        ignore_missing(iam.delete_role, RoleName=role)
        log("role and instance profile deleted")

    group_id = state.get("security_group_id")
    if group_id:
        ignore_missing(ec2.delete_security_group, GroupId=group_id)
        log("security group deleted")

    key_name = state.get("key_name")
    if key_name:
        ignore_missing(ec2.delete_key_pair, KeyName=key_name)
        log("key pair deleted from AWS; delete the local .pem yourself")

    bucket = state.get("bucket")
    if bucket and args.delete_bucket:
        empty_bucket(session, bucket)
        ignore_missing(session.client("s3").delete_bucket, Bucket=bucket)
        log(f"bucket {bucket} deleted")
    elif bucket:
        log(f"bucket {bucket} kept; pass --delete-bucket to remove it")

    args.state.unlink(missing_ok=True)
    log("done. Check Billing and Cost Management to confirm nothing is left running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
