# Backup Vault 생성
resource "aws_backup_vault" "fcmates_backup_vault" {
  name        = var.vault_name
  kms_key_arn = var.kms_key_arn

  tags = merge(var.tags, {
    Name = "${var.vault_name}"
  })
}

# Backup Plan 생성
resource "aws_backup_plan" "fcmates_backup_plan" {
  name = var.plan_name

  # 일일 백업 규칙
  rule {
    rule_name         = "${var.plan_name}-daily"
    target_vault_name = aws_backup_vault.fcmates_backup_vault.name
    schedule          = var.daily_backup_schedule
    start_window      = var.backup_start_window
    completion_window = var.backup_completion_window

    lifecycle {
      cold_storage_after = var.daily_cold_storage_after
      delete_after       = var.daily_delete_after
    }

    recovery_point_tags = merge(var.tags, {
      Name       = "${var.plan_name}-Daily-Backup"
      BackupType = "Daily"
    })
  }

  # 주간 백업 규칙 (장기 보관)
  rule {
    rule_name         = "${var.plan_name}-weekly"
    target_vault_name = aws_backup_vault.fcmates_backup_vault.name
    schedule          = var.weekly_backup_schedule
    start_window      = var.backup_start_window
    completion_window = var.backup_completion_window

    lifecycle {
      cold_storage_after = var.daily_cold_storage_after
      delete_after       = var.daily_delete_after
    }

    recovery_point_tags = merge(var.tags, {
      Name       = "${var.plan_name}-Weekly-Backup"
      BackupType = "Weekly"
    })
  }

  tags = merge(var.tags, {
    Name = var.plan_name
  })
}

# IAM Role for AWS Backup
resource "aws_iam_role" "backup_role" {
  name = "${var.plan_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "backup.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.plan_name}-Backup-Role"
  })
}

# AWS Backup Service Role Policy 연결
resource "aws_iam_role_policy_attachment" "backup_policy" {
  role       = aws_iam_role.backup_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForBackup"
}

# Backup Selection - EC2 및 S3 백업 설정
resource "aws_backup_selection" "fcmates_backup_selection" {
  iam_role_arn = aws_iam_role.backup_role.arn
  name         = "${var.plan_name}-selection"
  plan_id      = aws_backup_plan.fcmates_backup_plan.id

  # EC2 인스턴스 백업 (태그 기반)
  selection_tag {
    type  = "STRINGEQUALS"
    key   = "BackupEnabled"
    value = "true"
  }

  # 특정 리소스 ARN으로 백업 대상 지정
  resources = var.backup_resources

  # 백업에서 제외할 리소스
  not_resources = var.not_resources

  # 조건부 백업 설정
  dynamic "condition" {
    for_each = var.selection_conditions != null ? [var.selection_conditions] : []
    content {
      dynamic "string_equals" {
        for_each = condition.value.string_equals != null ? condition.value.string_equals : []
        content {
          key   = string_equals.value.key
          value = string_equals.value.value
        }
      }

      dynamic "string_like" {
        for_each = condition.value.string_like != null ? condition.value.string_like : []
        content {
          key   = string_like.value.key
          value = string_like.value.value
        }
      }

      dynamic "string_not_equals" {
        for_each = condition.value.string_not_equals != null ? condition.value.string_not_equals : []
        content {
          key   = string_not_equals.value.key
          value = string_not_equals.value.value
        }
      }

      dynamic "string_not_like" {
        for_each = condition.value.string_not_like != null ? condition.value.string_not_like : []
        content {
          key   = string_not_like.value.key
          value = string_not_like.value.value
        }
      }
    }
  }
}