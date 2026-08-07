"""Create the AWS resources the scoring service runs on.

One script, run from a laptop, that leaves behind a bucket holding the
model, an instance serving it and a role letting the second read the
first. Every step checks for what it is about to create, so running this
twice is the same as running it once.

    py deploy/aws/bootstrap.py --profile olist --bucket <globally-unique-name>

What it deliberately does not do: give the instance any write access to
the bucket, open any port other than 80 to the world, or place a
long-lived AWS key on the instance. The instance authenticates with a
role, which means there is no credential on the box to leak.

Teardown is the companion script, and running it returns the account to
empty.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = REPO_ROOT / "deploy" / "aws" / ".stack-state.json"
USER_DATA_PATH = Path(__file__).with_name("user_data.sh")

DEFAULT_REGION = "eu-west-2"
DEFAULT_REPO = "https://github.com/ariyan-tipu/olist-customer-retention"
DEFAULT_BRANCH = "main"
ROLE_NAME = "olist-scoring-instance"
PROFILE_NAME = "olist-scoring-instance"
SECURITY_GROUP = "olist-scoring-sg"
KEY_NAME = "olist-scoring-key"
NAMESPACE = "OlistRetention"

# Canonical publishes the current Ubuntu image id as a public SSM
# parameter. Reading it beats pasting an AMI id that goes stale, and it
# keeps the script correct in any region.
UBUNTU_SSM_PARAMETER = (
    "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
)


def log(message: str) -> None:
    print(f"[bootstrap] {message}", flush=True)


def detect_public_ip() -> str:
    """The address SSH will be limited to."""
    with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10) as response:
        return response.read().decode().strip()


def instance_policy(bucket: str) -> dict[str, Any]:
    """Read one prefix, write metrics in one namespace, nothing else."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ReadTheModelArtefact",
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket}/models/*"],
            },
            {
                "Sid": "ListOnlyTheModelPrefix",
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{bucket}"],
                "Condition": {"StringLike": {"s3:prefix": ["models/*"]}},
            },
            {
                "Sid": "PublishServiceMetrics",
                "Effect": "Allow",
                "Action": ["cloudwatch:PutMetricData"],
                "Resource": "*",
                "Condition": {"StringEquals": {"cloudwatch:namespace": NAMESPACE}},
            },
        ],
    }


class Stack:
    def __init__(self, session: boto3.Session, args: argparse.Namespace) -> None:
        self.session = session
        self.args = args
        self.region = session.region_name or DEFAULT_REGION
        self.s3 = session.client("s3")
        self.iam = session.client("iam")
        self.ec2 = session.client("ec2")
        self.ssm = session.client("ssm")
        self.state: dict[str, Any] = {"region": self.region, "bucket": args.bucket}

    # -- storage -------------------------------------------------------

    def ensure_bucket(self) -> None:
        bucket = self.args.bucket
        try:
            self.s3.head_bucket(Bucket=bucket)
            log(f"bucket {bucket} already exists")
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"404", "NoSuchBucket", "403"}:
                raise
            arguments: dict[str, Any] = {"Bucket": bucket}
            if self.region != "us-east-1":
                arguments["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
            self.s3.create_bucket(**arguments)
            log(f"created bucket {bucket}")

        self.s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        # Versioning is the cheap half of a rollback plan: an overwritten
        # model can be restored without a retrain.
        self.s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})
        self.s3.put_bucket_encryption(
            Bucket=bucket,
            ServerSideEncryptionConfiguration={
                "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
            },
        )
        log("bucket is private, versioned and encrypted at rest")

    # -- identity ------------------------------------------------------

    def ensure_role(self) -> str:
        trust = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        try:
            self.iam.create_role(
                RoleName=ROLE_NAME,
                AssumeRolePolicyDocument=json.dumps(trust),
                Description="Read the churn model from S3 and publish service metrics",
            )
            log(f"created role {ROLE_NAME}")
        except self.iam.exceptions.EntityAlreadyExistsException:
            log(f"role {ROLE_NAME} already exists")

        self.iam.put_role_policy(
            RoleName=ROLE_NAME,
            PolicyName="olist-scoring-least-privilege",
            PolicyDocument=json.dumps(instance_policy(self.args.bucket)),
        )

        try:
            self.iam.create_instance_profile(InstanceProfileName=PROFILE_NAME)
            log(f"created instance profile {PROFILE_NAME}")
        except self.iam.exceptions.EntityAlreadyExistsException:
            pass

        profile = self.iam.get_instance_profile(InstanceProfileName=PROFILE_NAME)
        attached = {role["RoleName"] for role in profile["InstanceProfile"]["Roles"]}
        if ROLE_NAME not in attached:
            self.iam.add_role_to_instance_profile(
                InstanceProfileName=PROFILE_NAME, RoleName=ROLE_NAME
            )
            # IAM is eventually consistent, and an instance launched the
            # instant after this call can fail to see the profile.
            log("waiting for the instance profile to propagate")
            time.sleep(15)
        self.state["role_name"] = ROLE_NAME
        self.state["instance_profile"] = PROFILE_NAME
        return PROFILE_NAME

    # -- network -------------------------------------------------------

    def ensure_key_pair(self) -> str:
        pem_path = Path(self.args.key_file or REPO_ROOT.parent / f"{KEY_NAME}.pem")
        try:
            self.ec2.describe_key_pairs(KeyNames=[KEY_NAME])
            log(f"key pair {KEY_NAME} already exists")
            if not pem_path.is_file():
                log(f"WARNING: no private key at {pem_path}; SSH will not work")
        except ClientError:
            created = self.ec2.create_key_pair(KeyName=KEY_NAME, KeyType="ed25519")
            pem_path.parent.mkdir(parents=True, exist_ok=True)
            pem_path.write_text(created["KeyMaterial"], encoding="utf-8")
            pem_path.chmod(0o600)
            log(f"created key pair, private key written to {pem_path}")
        self.state["key_name"] = KEY_NAME
        self.state["key_file"] = str(pem_path)
        return KEY_NAME

    def ensure_security_group(self) -> str:
        vpcs = self.ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
        if not vpcs["Vpcs"]:
            raise SystemExit("No default VPC in this region; create one or pass --vpc-id")
        vpc_id = self.args.vpc_id or vpcs["Vpcs"][0]["VpcId"]

        existing = self.ec2.describe_security_groups(
            Filters=[
                {"Name": "group-name", "Values": [SECURITY_GROUP]},
                {"Name": "vpc-id", "Values": [vpc_id]},
            ]
        )["SecurityGroups"]
        if existing:
            group_id = existing[0]["GroupId"]
            log(f"security group {SECURITY_GROUP} already exists")
        else:
            group_id = self.ec2.create_security_group(
                GroupName=SECURITY_GROUP,
                Description="Public HTTP to the scoring service, SSH from one address",
                VpcId=vpc_id,
            )["GroupId"]
            log(f"created security group {group_id}")

        ssh_cidr = self.args.ssh_cidr or f"{detect_public_ip()}/32"
        wanted = [
            {
                "IpProtocol": "tcp",
                "FromPort": 80,
                "ToPort": 80,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "public demo"}],
            },
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": ssh_cidr, "Description": "operator only"}],
            },
        ]
        for rule in wanted:
            try:
                self.ec2.authorize_security_group_ingress(GroupId=group_id, IpPermissions=[rule])
            except ClientError as error:
                if error.response["Error"]["Code"] != "InvalidPermission.Duplicate":
                    raise
        log(f"port 80 open to the internet, port 22 open to {ssh_cidr}")
        self.state["security_group_id"] = group_id
        return group_id

    # -- compute -------------------------------------------------------

    def latest_ubuntu_ami(self) -> str:
        ami = self.ssm.get_parameter(Name=UBUNTU_SSM_PARAMETER)["Parameter"]["Value"]
        log(f"using Ubuntu 24.04 image {ami}")
        return ami

    def render_user_data(self) -> str:
        model_uri = f"s3://{self.args.bucket}/models/churn_model.joblib"
        pip_spec = f"olist-retention[service,aws] @ git+{self.args.repo_url}@{self.args.branch}"
        text = USER_DATA_PATH.read_text(encoding="utf-8")
        replacements = {
            "@@PIP_SPEC@@": pip_spec,
            "@@MODEL_S3_URI@@": model_uri,
            "@@REGION@@": self.region,
        }
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        # Matching the placeholder shape rather than a bare "@@", which
        # the script's own header comment contains while explaining what
        # the placeholders are.
        remaining = re.findall(r"@@[A-Z_]+@@", text)
        if remaining:
            raise SystemExit(f"Unfilled placeholder in user data: {remaining[0]}")
        self.state["model_s3_uri"] = model_uri
        return text

    def existing_instance(self) -> dict[str, Any] | None:
        reservations = self.ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": ["olist-scoring"]},
                {"Name": "instance-state-name", "Values": ["pending", "running"]},
            ]
        )["Reservations"]
        for reservation in reservations:
            for instance in reservation["Instances"]:
                return instance
        return None

    def launch(self, profile_name: str, key_name: str, group_id: str) -> dict[str, Any]:
        running = self.existing_instance()
        if running is not None:
            log(f"instance {running['InstanceId']} is already running")
            return running

        response = self.ec2.run_instances(
            ImageId=self.latest_ubuntu_ami(),
            InstanceType=self.args.instance_type,
            MinCount=1,
            MaxCount=1,
            KeyName=key_name,
            SecurityGroupIds=[group_id],
            IamInstanceProfile={"Name": profile_name},
            UserData=self.render_user_data(),
            MetadataOptions={"HttpTokens": "required"},
            BlockDeviceMappings=[
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {"VolumeSize": 12, "VolumeType": "gp3", "DeleteOnTermination": True},
                }
            ],
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": "olist-scoring"},
                        {"Key": "Project", "Value": "olist-customer-retention"},
                    ],
                }
            ],
        )
        instance = response["Instances"][0]
        instance_id = instance["InstanceId"]
        log(f"launched {instance_id}, waiting for it to run")
        self.ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
        return self.ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0][
            "Instances"
        ][0]

    # -- orchestration -------------------------------------------------

    def run(self) -> int:
        self.ensure_bucket()
        profile_name = self.ensure_role()
        key_name = self.ensure_key_pair()
        group_id = self.ensure_security_group()
        instance = self.launch(profile_name, key_name, group_id)

        self.state["instance_id"] = instance["InstanceId"]
        self.state["public_ip"] = instance.get("PublicIpAddress", "")
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

        address = self.state["public_ip"]
        log("done")
        print()
        print(f"  instance   {self.state['instance_id']}")
        print(f"  address    http://{address}/")
        print(f"  readiness  http://{address}/readyz")
        print(f"  metrics    http://{address}/metrics")
        print(f"  state      {STATE_PATH}")
        print()
        print("The first boot installs the package and pulls the model, which takes a")
        print("few minutes. Publish a model first if you have not already:")
        print(
            f"  olist-retention publish --bucket {self.args.bucket} --profile {self.args.profile}"
        )
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default=None, help="Named AWS profile to use")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--bucket", required=True, help="Globally unique S3 bucket name")
    parser.add_argument("--instance-type", default="t3.micro")
    parser.add_argument("--vpc-id", default=None)
    parser.add_argument(
        "--ssh-cidr",
        default=None,
        help="Range allowed to reach port 22; defaults to this machine's address alone",
    )
    parser.add_argument("--key-file", default=None, help="Where to write the private key")
    parser.add_argument("--repo-url", default=DEFAULT_REPO)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    return Stack(session, args).run()


if __name__ == "__main__":
    sys.exit(main())
