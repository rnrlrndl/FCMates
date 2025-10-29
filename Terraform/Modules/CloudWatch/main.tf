locals {
  tags = {
    Terraform   = "true"
    Environment = "dev"
  }
}

# EC2 인스턴스 CPU 사용률 알람
resource "aws_cloudwatch_metric_alarm" "ec2_cpu_high" {
  alarm_name          = "${var.ec2_instance_name}-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.cpu_evaluation_periods
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = var.metric_period
  statistic           = "Average"
  threshold           = var.cpu_threshold
  alarm_description   = "This metric monitors EC2 CPU utilization"
  alarm_actions       = var.alarm_actions

  dimensions = {
    InstanceId = var.ec2_instance_id
  }

  tags = merge(local.tags, {
    Name = "${var.ec2_instance_name}-cpu-alarm"
  })
}

# EC2 인스턴스 메모리 사용률 알람 (CloudWatch Agent 필요)
resource "aws_cloudwatch_metric_alarm" "ec2_memory_high" {
  count               = var.enable_memory_alarm ? 1 : 0
  alarm_name          = "${var.ec2_instance_name}-memory-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.memory_evaluation_periods
  metric_name         = "mem_used_percent"
  namespace           = "CWAgent"
  period              = var.metric_period
  statistic           = "Average"
  threshold           = var.memory_threshold
  alarm_description   = "This metric monitors EC2 memory utilization"
  alarm_actions       = var.alarm_actions

  dimensions = {
    InstanceId = var.ec2_instance_id
  }

  tags = merge(local.tags, {
    Name = "${var.ec2_instance_name}-memory-alarm"
  })
}

# EC2 인스턴스 디스크 사용률 알람
resource "aws_cloudwatch_metric_alarm" "ec2_disk_high" {
  count               = var.enable_disk_alarm ? 1 : 0
  alarm_name          = "${var.ec2_instance_name}-disk-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.disk_evaluation_periods
  metric_name         = "disk_used_percent"
  namespace           = "CWAgent"
  period              = var.metric_period
  statistic           = "Average"
  threshold           = var.disk_threshold
  alarm_description   = "This metric monitors EC2 disk utilization"
  alarm_actions       = var.alarm_actions

  dimensions = {
    InstanceId = var.ec2_instance_id
    path       = "/"
    fstype     = "ext4"
  }

  tags = merge(local.tags, {
    Name = "${var.ec2_instance_name}-disk-alarm"
  })
}

# EC2 인스턴스 상태 체크 알람
resource "aws_cloudwatch_metric_alarm" "ec2_status_check_failed" {
  alarm_name          = "${var.ec2_instance_name}-status-check-failed"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "StatusCheckFailed"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Average"
  threshold           = 0
  alarm_description   = "This metric monitors EC2 status checks"
  alarm_actions       = var.alarm_actions

  dimensions = {
    InstanceId = var.ec2_instance_id
  }

  tags = merge(local.tags, {
    Name = "${var.ec2_instance_name}-status-check-alarm"
  })
}

# S3 버킷 객체 수 알람
resource "aws_cloudwatch_metric_alarm" "s3_object_count" {
  count               = var.enable_s3_alarm ? 1 : 0
  alarm_name          = "${var.s3_bucket_name}-object-count-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = var.s3_evaluation_periods
  metric_name         = "NumberOfObjects"
  namespace           = "AWS/S3"
  period              = 86400  # 24시간
  statistic           = "Average"
  threshold           = var.s3_object_count_threshold
  alarm_description   = "This metric monitors S3 object count"
  alarm_actions       = var.alarm_actions

  dimensions = {
    BucketName = var.s3_bucket_name
    StorageType = "AllStorageTypes"
  }

  tags = merge(local.tags, {
    Name = "${var.s3_bucket_name}-object-count-alarm"
  })
}

# S3 버킷 크기 알람
resource "aws_cloudwatch_metric_alarm" "s3_bucket_size" {
  count               = var.enable_s3_alarm ? 1 : 0
  alarm_name          = "${var.s3_bucket_name}-size-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.s3_evaluation_periods
  metric_name         = "BucketSizeBytes"
  namespace           = "AWS/S3"
  period              = 86400  # 24시간
  statistic           = "Average"
  threshold           = var.s3_bucket_size_threshold
  alarm_description   = "This metric monitors S3 bucket size"
  alarm_actions       = var.alarm_actions

  dimensions = {
    BucketName = var.s3_bucket_name
    StorageType = "StandardStorage"
  }

  tags = merge(local.tags, {
    Name = "${var.s3_bucket_name}-size-alarm"
  })
}

# CloudWatch 대시보드
resource "aws_cloudwatch_dashboard" "main" {
  count          = var.create_dashboard ? 1 : 0
  dashboard_name = var.dashboard_name

  dashboard_body = jsonencode({
    widgets = [
      # EC2 CPU 위젯
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/EC2", "CPUUtilization", { stat = "Average", label = "CPU Average" }]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "EC2 CPU Utilization"
          yAxis = {
            left = {
              min = 0
              max = 100
            }
          }
        }
      },
      # EC2 네트워크 위젯
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/EC2", "NetworkIn", { stat = "Sum", label = "Network In" }],
            [".", "NetworkOut", { stat = "Sum", label = "Network Out" }]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
          title  = "EC2 Network Traffic"
        }
      },
      # S3 버킷 크기 위젯
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/S3", "BucketSizeBytes", "BucketName", var.s3_bucket_name, "StorageType", "StandardStorage", { stat = "Average" }]
          ]
          period = 86400
          stat   = "Average"
          region = var.aws_region
          title  = "S3 Bucket Size"
        }
      },
      # S3 객체 수 위젯
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/S3", "NumberOfObjects", "BucketName", var.s3_bucket_name, "StorageType", "AllStorageTypes", { stat = "Average" }]
          ]
          period = 86400
          stat   = "Average"
          region = var.aws_region
          title  = "S3 Object Count"
        }
      }
    ]
  })
}

# CloudWatch Log Group (EC2 로그 수집용)
resource "aws_cloudwatch_log_group" "ec2_logs" {
  count             = var.create_log_group ? 1 : 0
  name              = "/aws/ec2/${var.ec2_instance_name}"
  retention_in_days = var.log_retention_days

  tags = merge(local.tags, {
    Name = "${var.ec2_instance_name}-log-group"
  })
}

# CloudWatch Log Stream
resource "aws_cloudwatch_log_stream" "ec2_log_stream" {
  count          = var.create_log_group ? 1 : 0
  name           = "${var.ec2_instance_name}-stream"
  log_group_name = aws_cloudwatch_log_group.ec2_logs[0].name
}
