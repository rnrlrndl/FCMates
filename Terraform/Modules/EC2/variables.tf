variable "subnet_id" {
    description = "EC2 인스턴스를 배치할 Subnet ID"
    type        = string
}

variable "ami_id" {
  description = "AMI ID for EC2 instance"
  type        = string
}