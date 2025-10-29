# EC2 기본 정보
output "ec2_instance_id" {
  description = "The ID of the EC2 instance"
  value       = module.ec2_instance.id
}

output "ec2_instance_arn" {
  description = "The ARN of the EC2 instance"
  value       = module.ec2_instance.arn
}

output "ec2_instance_state" {
  description = "The current state of the EC2 instance (running, stopped, etc.)"
  value       = module.ec2_instance.instance_state
}

output "ec2_availability_zone" {
  description = "The Availability Zone of the EC2 instance"
  value       = module.ec2_instance.availability_zone
}

# 네트워크 관련
output "ec2_public_ip" {
  description = "The public IP address assigned to the EC2 instance"
  value       = module.ec2_instance.public_ip
}

output "ec2_public_dns" {
  description = "The public DNS name assigned to the EC2 instance"
  value       = module.ec2_instance.public_dns
}

output "ec2_private_dns" {
  description = "The private DNS name assigned to the EC2 instance"
  value       = module.ec2_instance.private_dns
}

# IAM Role 관련
output "ec2_iam_role_name" {
  description = "The name of the IAM role attached to the EC2 instance"
  value       = module.ec2_instance.iam_role_name
}

output "ec2_iam_role_arn" {
  description = "The ARN of the IAM role attached to the EC2 instance"
  value       = module.ec2_instance.iam_role_arn
}

# EBS 볼륨 관련
output "ec2_root_block_device" {
  description = "The root block device information"
  value       = module.ec2_instance.root_block_device
}

# 태그
output "ec2_tags_all" {
  description = "All tags associated with the EC2 instance"
  value       = module.ec2_instance.tags_all
}
