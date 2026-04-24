variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "clinical-nlp-pipeline"
}

variable "environment" {
  description = "Environment tag (dev/staging/prod)"
  type        = string
  default     = "dev"
}

variable "emr_instance_type" {
  description = "EC2 instance type for EMR nodes"
  type        = string
  default     = "m5.xlarge"
}

variable "emr_worker_count" {
  description = "Number of EMR worker nodes"
  type        = number
  default     = 3
}

variable "emr_release" {
  description = "EMR release label"
  type        = string
  default     = "emr-7.0.0"
}
