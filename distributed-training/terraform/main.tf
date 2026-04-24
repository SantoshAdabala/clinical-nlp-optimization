###############################################################################
# Clinical NLP Distributed Pipeline — Infrastructure as Code
#
# Provisions:
# - S3 bucket (data lake + scripts + logs)
# - IAM roles for EMR (service role + EC2 instance profile)
# - Step Functions state machine (pipeline orchestration)
# - Security group for EMR cluster
###############################################################################

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
