variable "kms_key_id" {
  description = "KMS key ID for SNS topic encryption"
  type        = string

}

# 이메일 알림 설정 변수
variable "notification_email" {
  description = "Email address for SNS notifications"
  type        = string
  # 기본값 없음 - main.tf에서 반드시 입력해야 함
}
