###############################################################################
# S3 Bucket — Data Lake for pipeline input/output/scripts/logs
###############################################################################

resource "aws_s3_bucket" "pipeline" {
  bucket = "${var.project_name}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "02-distributed-training"
  }
}

resource "aws_s3_bucket_versioning" "pipeline" {
  bucket = aws_s3_bucket.pipeline.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "pipeline" {
  bucket = aws_s3_bucket.pipeline.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "pipeline" {
  bucket = aws_s3_bucket.pipeline.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "pipeline" {
  bucket = aws_s3_bucket.pipeline.id

  rule {
    id     = "cleanup-logs"
    status = "Enabled"

    filter {
      prefix = "logs/"
    }

    expiration {
      days = 30
    }
  }
}

# Upload pipeline scripts
resource "aws_s3_object" "spark_pipeline" {
  bucket = aws_s3_bucket.pipeline.id
  key    = "scripts/spark_pipeline.py"
  source = "${path.module}/../spark_pipeline.py"
  etag   = filemd5("${path.module}/../spark_pipeline.py")
}

resource "aws_s3_object" "bootstrap" {
  bucket = aws_s3_bucket.pipeline.id
  key    = "scripts/bootstrap.sh"
  source = "${path.module}/../bootstrap.sh"
  etag   = filemd5("${path.module}/../bootstrap.sh")
}
