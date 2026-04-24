###############################################################################
# IAM Roles — EMR Service Role + EC2 Instance Profile + Step Functions
###############################################################################

# --- EMR Service Role ---
resource "aws_iam_role" "emr_service" {
  name = "${var.project_name}-emr-service"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "elasticmapreduce.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project = var.project_name
  }
}

resource "aws_iam_role_policy_attachment" "emr_service" {
  role       = aws_iam_role.emr_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceRole"
}

# --- EMR EC2 Instance Role ---
resource "aws_iam_role" "emr_ec2" {
  name = "${var.project_name}-emr-ec2"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project = var.project_name
  }
}

resource "aws_iam_role_policy_attachment" "emr_ec2" {
  role       = aws_iam_role.emr_ec2.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceforEC2Role"
}

# S3 access for the pipeline bucket
resource "aws_iam_role_policy" "emr_ec2_s3" {
  name = "${var.project_name}-emr-s3-access"
  role = aws_iam_role.emr_ec2.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.pipeline.arn,
          "${aws_s3_bucket.pipeline.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${aws_s3_bucket.pipeline.arn}",
          "${aws_s3_bucket.pipeline.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "emr_ec2" {
  name = "${var.project_name}-emr-ec2-profile"
  role = aws_iam_role.emr_ec2.name
}

# --- Step Functions Execution Role ---
resource "aws_iam_role" "step_functions" {
  name = "${var.project_name}-stepfunctions"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "states.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project = var.project_name
  }
}

resource "aws_iam_role_policy" "step_functions" {
  name = "${var.project_name}-stepfunctions-policy"
  role = aws_iam_role.step_functions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "elasticmapreduce:RunJobFlow",
          "elasticmapreduce:DescribeCluster",
          "elasticmapreduce:TerminateJobFlows",
          "elasticmapreduce:AddJobFlowSteps",
          "elasticmapreduce:DescribeStep",
          "elasticmapreduce:CancelSteps",
          "elasticmapreduce:ListSteps"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = [
          aws_iam_role.emr_service.arn,
          aws_iam_role.emr_ec2.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "events:PutTargets",
          "events:PutRule",
          "events:DescribeRule"
        ]
        Resource = "arn:aws:events:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEMR*"
      }
    ]
  })
}
