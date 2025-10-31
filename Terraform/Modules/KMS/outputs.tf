output "key_id" {
  description = "KMS key ID for use by other modules"
  value       = module.kms.key_id
}

output "key_arn" {
  description = "KMS key ARN for reference"
  value       = module.kms.key_arn
}
