import argparse
import json
import time
import boto3
from pathlib import Path

RESULTS_DIR = Path("results")


def upload_scripts(s3_client, bucket, region):
    print("Uploading scripts to S3...")

    files_to_upload = {
        "spark_pipeline.py": "scripts/spark_pipeline.py",
        "spark_pipeline_emr.py": "scripts/spark_pipeline_emr.py",
        "bootstrap.sh": "scripts/bootstrap.sh",
    }

    for local_file, s3_key in files_to_upload.items():
        if Path(local_file).exists():
            s3_client.upload_file(local_file, bucket, s3_key)
            print(f"  Uploaded: s3://{bucket}/{s3_key}")
        else:
            print(f"  WARNING: {local_file} not found, skipping")


def create_emr_cluster(
    emr_client,
    bucket,
    region,
    instance_type="m5.xlarge",
    num_workers=3,
    sample_fraction=None,
):
    print("\nCreating EMR cluster...")

    spark_args = [
        "spark-submit",
        "--deploy-mode", "cluster",
        "--conf", "spark.sql.adaptive.enabled=true",
        "--conf", "spark.sql.adaptive.coalescePartitions.enabled=true",
        "--conf", "spark.sql.adaptive.skewJoin.enabled=true",
        "--conf", "spark.serializer=org.apache.spark.serializer.KryoSerializer",
        "--conf", "spark.dynamicAllocation.enabled=true",
        f"s3://{bucket}/scripts/spark_pipeline.py",
        "--input", f"s3://{bucket}/data/amazon_reviews/",
        "--output", f"s3://{bucket}/output/clinical-nlp-features/",
    ]

    if sample_fraction:
        spark_args.extend(["--sample", str(sample_fraction)])

    response = emr_client.run_job_flow(
        Name="ClinicalNLP-DistributedPipeline",
        ReleaseLabel="emr-7.0.0",
        LogUri=f"s3://{bucket}/logs/emr/",
        Instances={
            "MasterInstanceType": instance_type,
            "SlaveInstanceType": instance_type,
            "InstanceCount": num_workers + 1,
            "KeepJobFlowAliveWhenNoSteps": False,
            "Ec2SubnetId": "subnet-05e544c25bd9476e1",
        },
        Steps=[
            {
                "Name": "Run NLP Pipeline",
                "ActionOnFailure": "TERMINATE_CLUSTER",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": spark_args,
                },
            }
        ],
        Applications=[
            {"Name": "Spark"},
        ],
        BootstrapActions=[
            {
                "Name": "Install Dependencies",
                "ScriptBootstrapAction": {
                    "Path": f"s3://{bucket}/scripts/bootstrap.sh",
                },
            }
        ],
        Configurations=[
            {
                "Classification": "spark-defaults",
                "Properties": {
                    "spark.sql.shuffle.partitions": "200",
                    "spark.default.parallelism": "200",
                    "spark.sql.parquet.compression.codec": "snappy",
                },
            },
            {
                "Classification": "spark-env",
                "Configurations": [
                    {
                        "Classification": "export",
                        "Properties": {
                            "PYSPARK_PYTHON": "/usr/bin/python3",
                        },
                    }
                ],
            },
        ],
        ServiceRole="EMR_DefaultRole",
        JobFlowRole="EMR_EC2_DefaultRole",
        Tags=[
            {"Key": "Project", "Value": "ClinicalNLP-Portfolio"},
            {"Key": "Component", "Value": "02-distributed-training"},
        ],
        AutoTerminationPolicy={
            "IdleTimeout": 300,
        },
    )

    cluster_id = response["JobFlowId"]
    print(f"  Cluster ID: {cluster_id}")
    print(f"  Instance type: {instance_type}")
    print(f"  Workers: {num_workers}")

    return cluster_id


def monitor_cluster(emr_client, cluster_id):
    print("\nMonitoring cluster...")

    terminal_states = {"COMPLETED", "FAILED", "TERMINATED", "TERMINATED_WITH_ERRORS"}

    while True:
        response = emr_client.describe_cluster(ClusterId=cluster_id)
        state = response["Cluster"]["Status"]["State"]
        print(f"  Status: {state}")

        if state in terminal_states:
            break

        steps = emr_client.list_steps(ClusterId=cluster_id)
        for step in steps["Steps"]:
            step_state = step["Status"]["State"]
            print(f"    Step '{step['Name']}': {step_state}")

        time.sleep(30)

    final_state = response["Cluster"]["Status"]["State"]
    if final_state in {"TERMINATED_WITH_ERRORS", "FAILED"}:
        reason = response["Cluster"]["Status"].get("StateChangeReason", {})
        print(f"\n  FAILED: {reason.get('Message', 'Unknown error')}")
        return False

    print(f"\n  Cluster completed: {final_state}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Deploy Spark Pipeline to EMR")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--instance-type", default="m5.xlarge", help="EC2 instance type")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of worker nodes")
    parser.add_argument("--sample", type=float, default=None, help="Sample fraction (0-1)")
    parser.add_argument("--no-monitor", action="store_true", help="Don't wait for completion")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Deploying to EMR")
    print(f"  Bucket: {args.bucket}")
    print(f"  Region: {args.region}")

    session = boto3.Session(region_name=args.region)
    s3_client = session.client("s3")
    emr_client = session.client("emr")

    upload_scripts(s3_client, args.bucket, args.region)

    cluster_id = create_emr_cluster(
        emr_client,
        args.bucket,
        args.region,
        instance_type=args.instance_type,
        num_workers=args.num_workers,
        sample_fraction=args.sample,
    )

    deployment_info = {
        "cluster_id": cluster_id,
        "bucket": args.bucket,
        "region": args.region,
        "instance_type": args.instance_type,
        "num_workers": args.num_workers,
        "output_path": f"s3://{args.bucket}/output/clinical-nlp-features/",
    }

    with open(RESULTS_DIR / "deployment_info.json", "w") as f:
        json.dump(deployment_info, f, indent=2)

    if not args.no_monitor:
        success = monitor_cluster(emr_client, cluster_id)
        deployment_info["status"] = "success" if success else "failed"

        with open(RESULTS_DIR / "deployment_info.json", "w") as f:
            json.dump(deployment_info, f, indent=2)
    else:
        print(f"\n  Monitor with: aws emr describe-cluster --cluster-id {cluster_id}")


if __name__ == "__main__":
    main()
