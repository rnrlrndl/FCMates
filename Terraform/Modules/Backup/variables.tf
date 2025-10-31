variable "kms_key_arn" {
  description = "The server-side encryption key that is used to protect your backups"
  type        = string
  default     = null
}

variable "backup_resources" {
  description = "An array of strings that either contain Amazon Resource Names (ARNs) or match patterns of resources to assign to a backup plan"
  type        = list(string)
  default     = []
}

variable "not_resources" {
  description = "An array of strings that either contain Amazon Resource Names (ARNs) or match patterns of resources to exclude from a backup plan"
  type        = list(string)
  default     = []
}

variable "selection_conditions" {
  description = "An array of conditions used to specify a set of resources to assign to a backup plan"
  type = object({
    string_equals = optional(list(object({
      key   = string
      value = string
    })), [])
    string_like = optional(list(object({
      key   = string
      value = string
    })), [])
    string_not_equals = optional(list(object({
      key   = string
      value = string
    })), [])
    string_not_like = optional(list(object({
      key   = string
      value = string
    })), [])
  })
  default = null
}

variable "vault_name" {
  description = "Name of the backup vault"
  type        = string
  default     = "fcmates-backup-vault"
}

variable "plan_name" {
  description = "Name of the backup plan"
  type        = string
  default     = "fcmates-backup-plan"
}

variable "daily_backup_schedule" {
  description = "Cron schedule for daily backups"
  type        = string
  default     = "cron(0 2 ? * * *)" # 매일 오전 2시 (UTC)
}

variable "weekly_backup_schedule" {
  description = "Cron schedule for weekly backups"
  type        = string
  default     = "cron(0 3 ? * SUN *)" # 매주 일요일 오전 3시 (UTC)
}

variable "daily_cold_storage_after" {
  description = "Number of days after which daily backup transitions to cold storage"
  type        = number
  default     = 30
}

variable "daily_delete_after" {
  description = "Number of days after which daily backup is deleted"
  type        = number
  default     = 365
}

variable "weekly_cold_storage_after" {
  description = "Number of days after which weekly backup transitions to cold storage"
  type        = number
  default     = 90
}

variable "weekly_delete_after" {
  description = "Number of days after which weekly backup is deleted"
  type        = number
  default     = 2555 # 7 years
}

variable "backup_start_window" {
  description = "The amount of time in minutes before beginning a backup"
  type        = number
  default     = 60
}

variable "backup_completion_window" {
  description = "The amount of time in minutes AWS Backup attempts a backup before canceling the job and returning an error"
  type        = number
  default     = 300
}

variable "enable_continuous_backup" {
  description = "Enable continuous backup for point-in-time recovery"
  type        = bool
  default     = false
}

variable "tags" {
  description = "A map of tags to assign to the resources"
  type        = map(string)
  default = {
    Environment = "dev"
    Project     = "FCMates"
  }
}