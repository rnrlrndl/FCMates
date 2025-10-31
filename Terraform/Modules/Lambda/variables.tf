# 기본 설정 변수들
variable "name_prefix" {
  description = "Lambda 리소스들의 이름 접두사"
  type        = string
  default     = "fcmates"
}

variable "aws_region" {
  description = "AWS 리전"
  type        = string
  default     = "ap-northeast-2"
}

variable "tags" {
  description = "모든 리소스에 적용될 태그"
  type        = map(string)
  default = {
    Terraform   = "true"
    Environment = "dev"
    Project     = "FCMates"
    Component   = "Lambda"
  }
}

# 연동할 다른 AWS 서비스 정보
variable "ec2_instance_id" {
  description = "모니터링할 EC2 인스턴스 ID"
  type        = string
}

variable "s3_bucket_name" {
  description = "정리할 S3 버킷 이름"
  type        = string
  default     = "fcmates-bucket"
}

variable "sns_topic_arn" {
  description = "알림을 보낼 SNS 토픽 ARN"
  type        = string
}

# Lambda 함수 설정
variable "lambda_functions" {
  description = "생성할 Lambda 함수들의 설정"
  type = map(object({
    source_code             = string
    timeout                 = number
    environment_variables   = map(string)
  }))
  default = {}
}

# 스케줄 설정
variable "s3_cleanup_schedule" {
  description = "S3 정리 작업 스케줄 (cron 표현식)"
  type        = string
  default     = "cron(0 6 * * ? *)"  # 매일 오전 6시 (UTC)
}

# 로그 보존 기간
variable "log_retention_days" {
  description = "Lambda 함수 로그 보존 기간 (일)"
  type        = number
  default     = 14
}

# Auto Scaling 설정 (선택적)
variable "autoscaling_group_names" {
  description = "모니터링할 Auto Scaling Group 이름들"
  type        = list(string)
  default     = []
}

# 복구 설정
variable "enable_ec2_auto_recovery" {
  description = "EC2 자동 복구 활성화 여부"
  type        = bool
  default     = true
}

variable "enable_s3_cleanup" {
  description = "S3 자동 정리 활성화 여부"
  type        = bool
  default     = true
}

variable "s3_cleanup_days" {
  description = "S3에서 삭제할 파일의 나이 (일)"
  type        = number
  default     = 30
}

# CloudWatch 메트릭 임계값
variable "cpu_threshold" {
  description = "CPU 사용률 복구 임계값 (%)"
  type        = number
  default     = 90
}

variable "memory_threshold" {
  description = "메모리 사용률 복구 임계값 (%)"
  type        = number
  default     = 90
}

# 알림 설정
variable "notification_email" {
  description = "복구 작업 알림을 받을 이메일 주소"
  type        = string
  default     = ""
}