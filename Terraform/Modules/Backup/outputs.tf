output "backup_vault_id" {
  description = "The unique identifier of the backup vault"
  value       = aws_backup_vault.fcmates_backup_vault.id
}

output "backup_vault_arn" {
  description = "The ARN of the backup vault"
  value       = aws_backup_vault.fcmates_backup_vault.arn
}

output "backup_vault_name" {
  description = "The name of the backup vault"
  value       = aws_backup_vault.fcmates_backup_vault.name
}

output "backup_plan_id" {
  description = "The unique identifier of the backup plan"
  value       = aws_backup_plan.fcmates_backup_plan.id
}

output "backup_plan_arn" {
  description = "The ARN of the backup plan"
  value       = aws_backup_plan.fcmates_backup_plan.arn
}

output "backup_plan_version" {
  description = "The version of the backup plan"
  value       = aws_backup_plan.fcmates_backup_plan.version
}

output "backup_selection_id" {
  description = "The unique identifier of the backup selection"
  value       = aws_backup_selection.fcmates_backup_selection.id
}

output "backup_role_arn" {
  description = "The ARN of the IAM role used by AWS Backup"
  value       = aws_iam_role.backup_role.arn
}

output "backup_role_name" {
  description = "The name of the IAM role used by AWS Backup"
  value       = aws_iam_role.backup_role.name
}