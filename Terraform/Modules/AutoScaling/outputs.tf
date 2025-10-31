output "autoscaling_group_id" {
  description = "The auto scaling group id"
  value       = module.autoscaling.autoscaling_group_id
}

output "autoscaling_group_name" {
  description = "The auto scaling group name"
  value       = module.autoscaling.autoscaling_group_name
}

output "autoscaling_group_arn" {
  description = "The ARN for this auto scaling group"
  value       = module.autoscaling.autoscaling_group_arn
}

output "autoscaling_group_min_size" {
  description = "The minimum size of the auto scaling group"
  value       = module.autoscaling.autoscaling_group_min_size
}

output "autoscaling_group_max_size" {
  description = "The maximum size of the auto scaling group"
  value       = module.autoscaling.autoscaling_group_max_size
}

output "autoscaling_group_desired_capacity" {
  description = "The number of Amazon EC2 instances that should be running in the group"
  value       = module.autoscaling.autoscaling_group_desired_capacity
}

output "launch_template_id" {
  description = "The ID of the launch template"
  value       = module.autoscaling.launch_template_id
}

output "launch_template_arn" {
  description = "The ARN of the launch template"
  value       = module.autoscaling.launch_template_arn
}

output "launch_template_name" {
  description = "The name of the launch template"
  value       = module.autoscaling.launch_template_name
}

output "launch_template_latest_version" {
  description = "The latest version of the launch template"
  value       = module.autoscaling.launch_template_latest_version
}

output "iam_role_name" {
  description = "The name of the IAM role"
  value       = try(module.autoscaling.iam_role_name, null)
}

output "iam_role_arn" {
  description = "The Amazon Resource Name (ARN) specifying the IAM role"
  value       = try(module.autoscaling.iam_role_arn, null)
}

output "iam_instance_profile_arn" {
  description = "ARN assigned by AWS to the instance profile"
  value       = try(module.autoscaling.iam_instance_profile_arn, null)
}

output "iam_instance_profile_id" {
  description = "ID assigned to the instance profile"
  value       = try(module.autoscaling.iam_instance_profile_id, null)
}

output "security_group_id" {
  description = "ID of the security group"
  value       = aws_security_group.autoscaling_sg.id
}

output "security_group_arn" {
  description = "ARN of the security group"
  value       = aws_security_group.autoscaling_sg.arn
}