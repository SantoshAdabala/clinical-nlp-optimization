###############################################################################
# Step Functions State Machine — Pipeline Orchestration
###############################################################################

resource "aws_sfn_state_machine" "pipeline" {
  name     = "${var.project_name}-orchestration"
  role_arn = aws_iam_role.step_functions.arn

  definition = jsonencode({
    Comment = "Clinical NLP Distributed Pipeline"
    StartAt = "CreateEMRCluster"
    States = {
      CreateEMRCluster = {
        Type     = "Task"
        Resource = "arn:aws:states:::elasticmapreduce:createCluster.sync"
        Parameters = {
          Name         = "${var.project_name}-cluster"
          ReleaseLabel = var.emr_release
          LogUri       = "s3://${aws_s3_bucket.pipeline.id}/logs/emr/"
          Instances = {
            MasterInstanceType        = var.emr_instance_type
            SlaveInstanceType         = var.emr_instance_type
            InstanceCount             = var.emr_worker_count + 1
            KeepJobFlowAliveWhenNoSteps = true
            EmrManagedMasterSecurityGroup = aws_security_group.emr.id
            EmrManagedSlaveSecurityGroup  = aws_security_group.emr.id
          }
          Applications = [{ Name = "Spark" }]
          BootstrapActions = [
            {
              Name = "InstallDeps"
              ScriptBootstrapAction = {
                Path = "s3://${aws_s3_bucket.pipeline.id}/scripts/bootstrap.sh"
              }
            }
          ]
          Configurations = [
            {
              Classification = "spark-defaults"
              Properties = {
                "spark.sql.adaptive.enabled"                       = "true"
                "spark.sql.adaptive.coalescePartitions.enabled"    = "true"
                "spark.sql.adaptive.skewJoin.enabled"              = "true"
                "spark.serializer"                                 = "org.apache.spark.serializer.KryoSerializer"
                "spark.sql.shuffle.partitions"                     = "200"
              }
            }
          ]
          ServiceRole = aws_iam_role.emr_service.arn
          JobFlowRole = aws_iam_instance_profile.emr_ec2.arn
          AutoTerminationPolicy = { IdleTimeout = 300 }
          Tags = [
            { Key = "Project", Value = var.project_name },
            { Key = "ManagedBy", Value = "Terraform+StepFunctions" }
          ]
        }
        ResultPath = "$.cluster"
        Next       = "SubmitSparkJob"
      }

      SubmitSparkJob = {
        Type     = "Task"
        Resource = "arn:aws:states:::elasticmapreduce:addStep.sync"
        Parameters = {
          "ClusterId.$" = "$.cluster.ClusterId"
          Step = {
            Name            = "RunNLPPipeline"
            ActionOnFailure = "CONTINUE"
            HadoopJarStep = {
              Jar = "command-runner.jar"
              Args = [
                "spark-submit",
                "--deploy-mode", "cluster",
                "--conf", "spark.sql.adaptive.enabled=true",
                "--conf", "spark.dynamicAllocation.enabled=true",
                "s3://${aws_s3_bucket.pipeline.id}/scripts/spark_pipeline.py",
                "--input", "s3://${aws_s3_bucket.pipeline.id}/data/amazon_reviews/",
                "--output", "s3://${aws_s3_bucket.pipeline.id}/output/clinical-nlp-features/"
              ]
            }
          }
        }
        ResultPath = "$.step"
        Next       = "TerminateCluster"
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "TerminateOnFailure"
            ResultPath  = "$.error"
          }
        ]
      }

      TerminateCluster = {
        Type     = "Task"
        Resource = "arn:aws:states:::elasticmapreduce:terminateCluster.sync"
        Parameters = {
          "ClusterId.$" = "$.cluster.ClusterId"
        }
        Next = "PipelineSucceeded"
      }

      PipelineSucceeded = {
        Type = "Succeed"
      }

      TerminateOnFailure = {
        Type     = "Task"
        Resource = "arn:aws:states:::elasticmapreduce:terminateCluster.sync"
        Parameters = {
          "ClusterId.$" = "$.cluster.ClusterId"
        }
        Next = "PipelineFailed"
      }

      PipelineFailed = {
        Type  = "Fail"
        Error = "PipelineExecutionFailed"
        Cause = "Spark job failed — check EMR logs in S3"
      }
    }
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}
