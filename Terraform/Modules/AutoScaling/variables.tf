variable "name_prefix" {
  description = "Name prefix for all resources"
  type        = string
  default     = "fcmates"
}

variable "vpc_id" {
  description = "VPC ID where the auto scaling group will be created"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for the auto scaling group"
  type        = list(string)
}

variable "min_size" {
  description = "Minimum number of instances in the auto scaling group"
  type        = number
  default     = 1
}

variable "max_size" {
  description = "Maximum number of instances in the auto scaling group"
  type        = number
  default     = 3
}

variable "desired_capacity" {
  description = "Desired number of instances in the auto scaling group"
  type        = number
  default     = 2
}

variable "instance_type" {
  description = "Instance type for the auto scaling group"
  type        = string
  default     = "t3.micro"
}

variable "ami_id" {
  description = "AMI ID for the instances. If not provided, latest Amazon Linux 2 will be used"
  type        = string
  default     = null
}

variable "key_name" {
  description = "Key pair name for SSH access"
  type        = string
  default     = "FCMates"
}

variable "ebs_optimized" {
  description = "Enable EBS optimization for instances"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable detailed monitoring for instances"
  type        = bool
  default     = true
}

variable "root_volume_size" {
  description = "Size of the root volume in GB"
  type        = number
  default     = 20
}

variable "target_cpu_utilization" {
  description = "Target CPU utilization for auto scaling"
  type        = number
  default     = 70
}

variable "ssh_cidr_blocks" {
  description = "CIDR blocks allowed for SSH access"
  type        = string
  default     = "10.0.0.0/16"
}

variable "target_group_arns" {
  description = "List of target group ARNs for load balancer integration"
  type        = list(string)
  default     = []
}

variable "enable_notifications" {
  description = "Enable SNS notifications for auto scaling events"
  type        = bool
  default     = true
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for auto scaling notifications"
  type        = string
  default     = ""
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "tags" {
  description = "A map of tags to assign to the resources"
  type        = map(string)
  default = {
    Environment = "dev"
    Project     = "FCMates"
    Component   = "AutoScaling"
  }
}