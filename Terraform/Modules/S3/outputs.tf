# S3 Bucket Outputs (for main backup bucket)
# S3 버킷 이름(ID)
output "s3_bucket_id" {
  description = "The name of the main S3 backup bucket."
  value       = module.s3_bucket.s3_bucket_id
}

# S3 버킷 ARN (IAM 정책 등에서 참조 가능)
output "s3_bucket_arn" {
  description = "The ARN of the S3 backup bucket (used for IAM and Backup integration)."
  value       = module.s3_bucket.s3_bucket_arn
}

# S3 버킷 리전별 도메인명 (EC2에서 직접 접근 시 유용)
output "s3_bucket_regional_domain_name" {
  description = "The regional domain name of the S3 backup bucket."
  value       = module.s3_bucket.s3_bucket_bucket_regional_domain_name
}

# 버킷이 위치한 리전
output "s3_bucket_region" {
  description = "The AWS region where the S3 bucket resides."
  value       = module.s3_bucket.s3_bucket_region
}

# 라이프사이클 구성 확인용 (버전 관리 / Glacier 이동 등 확인)
output "s3_bucket_lifecycle_rules" {
  description = "The lifecycle rules applied to the S3 bucket (transition to IA, Glacier, etc.)."
  value       = module.s3_bucket.s3_bucket_lifecycle_configuration_rules
}

# 버킷 정책 (보안 정책 및 암호화 설정 검증용)
output "s3_bucket_policy" {
  description = "The policy applied to the S3 bucket, ensuring encryption and secure transport."
  value       = module.s3_bucket.s3_bucket_policy
}

# Log Bucket Outputs (for access log storage)
# 로그 버킷 이름
output "log_bucket_id" {
  description = "The name of the log bucket used to store S3 access logs."
  value       = module.log_bucket.s3_bucket_id
}

# 로그 버킷 ARN
output "log_bucket_arn" {
  description = "The ARN of the log bucket used for access logs."
  value       = module.log_bucket.s3_bucket_arn
}
