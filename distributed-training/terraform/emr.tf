###############################################################################
# EMR Security Group
###############################################################################

resource "aws_security_group" "emr" {
  name        = "${var.project_name}-emr-sg"
  description = "Security group for EMR cluster"

  # Allow all traffic within the security group (EMR nodes communicate)
  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project = var.project_name
  }
}
