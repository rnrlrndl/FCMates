# 현재 AWS 계정 정보 가져오기
data "aws_caller_identity" "current" {}

# Lambda 함수를 위한 IAM 역할
resource "aws_iam_role" "lambda_role" {
  name = "${var.name_prefix}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# Lambda 실행을 위한 기본 정책 연결
resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  role       = aws_iam_role.lambda_role.name
}

# EC2 및 CloudWatch 작업을 위한 추가 정책
resource "aws_iam_role_policy" "lambda_custom_policy" {
  name = "${var.name_prefix}-lambda-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:StartInstances",
          "ec2:StopInstances",
          "ec2:RebootInstances",
          "ec2:DescribeInstanceStatus",
          "ec2:CreateSnapshot",
          "ec2:DescribeSnapshots",
          "ec2:CreateTags"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:DeleteObject",
          "s3:PutObject",
          "s3:GetBucketVersioning",
          "s3:ListBucketVersions"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_name}",
          "arn:aws:s3:::${var.s3_bucket_name}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:DescribeAlarms"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish",
          "sns:GetTopicAttributes",
          "sns:ListTopics"
        ]
        Resource = var.sns_topic_arn
      },
      {
        Effect = "Allow"
        Action = [
          "autoscaling:DescribeAutoScalingGroups",
          "autoscaling:SetDesiredCapacity",
          "autoscaling:UpdateAutoScalingGroup"
        ]
        Resource = "*"
      }
    ]
  })
}

# Lambda 함수 코드를 위한 ZIP 파일 생성
data "archive_file" "lambda_zip" {
  for_each = var.lambda_functions
  
  type        = "zip"
  output_path = "${path.module}/lambda_${each.key}.zip"
  
  source {
    content = each.value.source_code
    filename = "lambda_function.py"
  }
}

# EC2 복구를 위한 Lambda 함수
resource "aws_lambda_function" "recovery_functions" {
  for_each = var.lambda_functions

  filename         = data.archive_file.lambda_zip[each.key].output_path
  function_name    = "${var.name_prefix}-${each.key}"
  role            = aws_iam_role.lambda_role.arn
  handler         = "lambda_function.lambda_handler"
  source_code_hash = data.archive_file.lambda_zip[each.key].output_base64sha256
  runtime         = "python3.9"
  timeout         = each.value.timeout

  environment {
    variables = merge(each.value.environment_variables, {
      EC2_INSTANCE_ID = var.ec2_instance_id
      S3_BUCKET_NAME  = var.s3_bucket_name
      SNS_TOPIC_ARN   = var.sns_topic_arn
      # AWS_REGION      = var.aws_region
    })
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-${each.key}"
  })
}

# CloudWatch 알람에서 Lambda 함수 호출을 위한 권한
resource "aws_lambda_permission" "allow_cloudwatch" {
  for_each = var.lambda_functions

  statement_id  = "AllowExecutionFromCloudWatch-${each.key}"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.recovery_functions[each.key].function_name
  principal     = "events.amazonaws.com"
  source_arn    = "arn:aws:events:${var.aws_region}:${data.aws_caller_identity.current.account_id}:rule/*"
}

# EventBridge 규칙 - CloudWatch 알람 상태 변경 감지
resource "aws_cloudwatch_event_rule" "alarm_state_change" {
  name        = "${var.name_prefix}-alarm-state-change"
  description = "Capture CloudWatch alarm state changes"

  event_pattern = jsonencode({
    source      = ["aws.cloudwatch"]
    detail-type = ["CloudWatch Alarm State Change"]
    detail = {
      state = {
        value = ["ALARM"]
      }
    }
  })

  tags = var.tags
}

# EventBridge 대상 - EC2 복구 Lambda 함수
resource "aws_cloudwatch_event_target" "ec2_recovery_target" {
  count = contains(keys(var.lambda_functions), "ec2-recovery") ? 1 : 0
  
  rule      = aws_cloudwatch_event_rule.alarm_state_change.name
  target_id = "EC2RecoveryTarget"
  arn       = aws_lambda_function.recovery_functions["ec2-recovery"].arn
}

# S3 정리를 위한 스케줄된 이벤트 (매일 자정)
resource "aws_cloudwatch_event_rule" "s3_cleanup_schedule" {
  count = contains(keys(var.lambda_functions), "s3-cleanup") ? 1 : 0
  
  name                = "${var.name_prefix}-s3-cleanup-schedule"
  description         = "Daily S3 cleanup schedule"
  schedule_expression = var.s3_cleanup_schedule

  tags = var.tags
}

# S3 정리 Lambda 함수 대상
resource "aws_cloudwatch_event_target" "s3_cleanup_target" {
  count = contains(keys(var.lambda_functions), "s3-cleanup") ? 1 : 0
  
  rule      = aws_cloudwatch_event_rule.s3_cleanup_schedule[0].name
  target_id = "S3CleanupTarget"
  arn       = aws_lambda_function.recovery_functions["s3-cleanup"].arn
}

# Lambda 함수 호출을 위한 EventBridge 권한
resource "aws_lambda_permission" "allow_eventbridge" {
  for_each = var.lambda_functions

  statement_id  = "AllowExecutionFromEventBridge-${each.key}"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.recovery_functions[each.key].function_name
  principal     = "events.amazonaws.com"
  source_arn    = each.key == "s3-cleanup" ? (
    contains(keys(var.lambda_functions), "s3-cleanup") ? 
    aws_cloudwatch_event_rule.s3_cleanup_schedule[0].arn : 
    aws_cloudwatch_event_rule.alarm_state_change.arn
  ) : aws_cloudwatch_event_rule.alarm_state_change.arn
}

# Lambda 함수 로그 그룹
resource "aws_cloudwatch_log_group" "lambda_logs" {
  for_each = var.lambda_functions

  name              = "/aws/lambda/${aws_lambda_function.recovery_functions[each.key].function_name}"
  retention_in_days = var.log_retention_days

  tags = var.tags
}