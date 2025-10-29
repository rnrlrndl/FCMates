# EC2 관련 변수
variable "ec2_instance_id" {
  description = "The ID of the EC2 instance to monitor"
  type        = string
}

variable "ec2_instance_name" {
  description = "The name of the EC2 instance for alarm naming"
  type        = string
  default     = "FCMates-EC2"
}

# S3 관련 변수
variable "s3_bucket_name" {
  description = "The name of the S3 bucket to monitor"
  type        = string
  default     = ""
}

# CPU 알람 설정
variable "cpu_threshold" {
  description = "The threshold for CPU utilization alarm (percentage)"
  type        = number
  default     = 80
}

variable "cpu_evaluation_periods" {
  description = "The number of periods to evaluate for CPU alarm"
  type        = number
  default     = 2
}

# 메모리 알람 설정
variable "memory_threshold" {
  description = "The threshold for memory utilization alarm (percentage)"
  type        = number
  default     = 80
}

variable "memory_evaluation_periods" {
  description = "The number of periods to evaluate for memory alarm"
  type        = number
  default     = 2
}

variable "enable_memory_alarm" {
  description = "Enable memory utilization alarm (requires CloudWatch Agent)"
  type        = bool
  default     = false
}

# 디스크 알람 설정
variable "disk_threshold" {
  description = "The threshold for disk utilization alarm (percentage)"
  type        = number
  default     = 80
}

variable "disk_evaluation_periods" {
  description = "The number of periods to evaluate for disk alarm"
  type        = number
  default     = 2
}

variable "enable_disk_alarm" {
  description = "Enable disk utilization alarm (requires CloudWatch Agent)"
  type        = bool
  default     = false
}

# S3 알람 설정
variable "s3_object_count_threshold" {
  description = "The minimum threshold for S3 object count alarm"
  type        = number
  default     = 1
}

variable "s3_bucket_size_threshold" {
  description = "The threshold for S3 bucket size alarm (bytes)"
  type        = number
  default     = 107374182400  # 100 GB
}

variable "s3_evaluation_periods" {
  description = "The number of periods to evaluate for S3 alarms"
  type        = number
  default     = 1
}

variable "enable_s3_alarm" {
  description = "Enable S3 bucket monitoring alarms"
  type        = bool
  default     = true
}

# 일반 설정
variable "metric_period" {
  description = "The period in seconds over which the metric is applied"
  type        = number
  default     = 300  # 5분
}

variable "alarm_actions" {
  description = "The list of actions to execute when this alarm transitions into an ALARM state (e.g., SNS topic ARN)"
  type        = list(string)
  default     = []
}

variable "aws_region" {
  description = "The AWS region for CloudWatch resources"
  type        = string
  default     = "ap-northeast-2"
}

# 대시보드 설정
variable "create_dashboard" {
  description = "Whether to create CloudWatch dashboard"
  type        = bool
  default     = true
}

variable "dashboard_name" {
  description = "The name of the CloudWatch dashboard"
  type        = string
  default     = "FCMates-Dashboard"
}

# 로그 그룹 설정
variable "create_log_group" {
  description = "Whether to create CloudWatch log group"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "The number of days to retain log events"
  type        = number
  default     = 7
}
