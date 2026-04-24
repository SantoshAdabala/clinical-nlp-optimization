output "s3_bucket" {
  description = "S3 bucket for pipeline data"
  value       = aws_s3_bucket.pipeline.id
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.pipeline.arn
}

output "step_functions_arn" {
  description = "Step Functions state machine ARN"
  value       = aws_sfn_state_machine.pipeline.arn
}

output "emr_service_role_arn" {
  description = "EMR service role ARN"
  value       = aws_iam_role.emr_service.arn
}

output "emr_ec2_instance_profile_arn" {
  description = "EMR EC2 instance profile ARN"
  value       = aws_iam_instance_profile.emr_ec2.arn
}

output "deploy_command" {
  description = "Command to run the pipeline via Step Functions"
  value       = "aws stepfunctions start-execution --state-machine-arn ${aws_sfn_state_machine.pipeline.arn} --input '{\"bucket\": \"${aws_s3_bucket.pipeline.id}\", \"instance_type\": \"${var.emr_instance_type}\", \"instance_count\": ${var.emr_worker_count + 1}}'"
}
