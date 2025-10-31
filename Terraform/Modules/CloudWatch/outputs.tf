# EC2 알람 관련 출력
output "ec2_cpu_alarm_arn" {
  description = "The ARN of the EC2 CPU utilization alarm"
  value       = aws_cloudwatch_metric_alarm.ec2_cpu_high.arn
}

output "ec2_cpu_alarm_name" {
  description = "The name of the EC2 CPU utilization alarm"
  value       = aws_cloudwatch_metric_alarm.ec2_cpu_high.alarm_name
}

output "ec2_status_check_alarm_arn" {
  description = "The ARN of the EC2 status check alarm"
  value       = aws_cloudwatch_metric_alarm.ec2_status_check_failed.arn
}

output "ec2_status_check_alarm_name" {
  description = "The name of the EC2 status check alarm"
  value       = aws_cloudwatch_metric_alarm.ec2_status_check_failed.alarm_name
}

output "ec2_memory_alarm_arn" {
  description = "The ARN of the EC2 memory utilization alarm (if enabled)"
  value       = var.enable_memory_alarm ? aws_cloudwatch_metric_alarm.ec2_memory_high[0].arn : null
}

output "ec2_disk_alarm_arn" {
  description = "The ARN of the EC2 disk utilization alarm (if enabled)"
  value       = var.enable_disk_alarm ? aws_cloudwatch_metric_alarm.ec2_disk_high[0].arn : null
}

# S3 알람 관련 출력
output "s3_object_count_alarm_arn" {
  description = "The ARN of the S3 object count alarm (if enabled)"
  value       = var.enable_s3_alarm ? aws_cloudwatch_metric_alarm.s3_object_count[0].arn : null
}

output "s3_bucket_size_alarm_arn" {
  description = "The ARN of the S3 bucket size alarm (if enabled)"
  value       = var.enable_s3_alarm ? aws_cloudwatch_metric_alarm.s3_bucket_size[0].arn : null
}

# 대시보드 관련 출력
output "dashboard_arn" {
  description = "The ARN of the CloudWatch dashboard (if created)"
  value       = var.create_dashboard ? aws_cloudwatch_dashboard.main[0].dashboard_arn : null
}

output "dashboard_name" {
  description = "The name of the CloudWatch dashboard (if created)"
  value       = var.create_dashboard ? var.dashboard_name : null
}

# 로그 그룹 관련 출력
output "log_group_name" {
  description = "The name of the CloudWatch log group (if created)"
  value       = var.create_log_group ? aws_cloudwatch_log_group.ec2_logs[0].name : null
}

output "log_group_arn" {
  description = "The ARN of the CloudWatch log group (if created)"
  value       = var.create_log_group ? aws_cloudwatch_log_group.ec2_logs[0].arn : null
}

output "log_stream_name" {
  description = "The name of the CloudWatch log stream (if created)"
  value       = var.create_log_group ? aws_cloudwatch_log_stream.ec2_log_stream[0].name : null
}

# 전체 알람 목록
output "all_alarm_arns" {
  description = "List of all CloudWatch alarm ARNs"
  value = compact([
    aws_cloudwatch_metric_alarm.ec2_cpu_high.arn,
    aws_cloudwatch_metric_alarm.ec2_status_check_failed.arn,
    var.enable_memory_alarm ? aws_cloudwatch_metric_alarm.ec2_memory_high[0].arn : null,
    var.enable_disk_alarm ? aws_cloudwatch_metric_alarm.ec2_disk_high[0].arn : null,
    var.enable_s3_alarm ? aws_cloudwatch_metric_alarm.s3_object_count[0].arn : null,
    var.enable_s3_alarm ? aws_cloudwatch_metric_alarm.s3_bucket_size[0].arn : null,
  ])
}