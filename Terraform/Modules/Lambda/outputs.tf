# Lambda 함수 ARNs
output "lambda_function_arns" {
  description = "Lambda 함수들의 ARN"
  value       = { for k, v in aws_lambda_function.recovery_functions : k => v.arn }
}

output "lambda_function_names" {
  description = "Lambda 함수들의 이름"
  value       = { for k, v in aws_lambda_function.recovery_functions : k => v.function_name }
}

# IAM 역할 정보
output "lambda_role_arn" {
  description = "Lambda 실행 역할의 ARN"
  value       = aws_iam_role.lambda_role.arn
}

output "lambda_role_name" {
  description = "Lambda 실행 역할의 이름"
  value       = aws_iam_role.lambda_role.name
}

# EventBridge 규칙 정보
output "alarm_state_change_rule_arn" {
  description = "CloudWatch 알람 상태 변경 이벤트 규칙 ARN"
  value       = aws_cloudwatch_event_rule.alarm_state_change.arn
}

output "s3_cleanup_rule_arn" {
  description = "S3 정리 스케줄 규칙 ARN"
  value       = length(aws_cloudwatch_event_rule.s3_cleanup_schedule) > 0 ? aws_cloudwatch_event_rule.s3_cleanup_schedule[0].arn : null
}

# CloudWatch 로그 그룹 정보
output "lambda_log_groups" {
  description = "Lambda 함수들의 CloudWatch 로그 그룹"
  value       = { for k, v in aws_cloudwatch_log_group.lambda_logs : k => v.name }
}

# Lambda 함수 상세 정보
output "lambda_functions_info" {
  description = "Lambda 함수들의 상세 정보"
  value = {
    for k, v in aws_lambda_function.recovery_functions : k => {
      arn           = v.arn
      function_name = v.function_name
      runtime       = v.runtime
      timeout       = v.timeout
      last_modified = v.last_modified
    }
  }
}

# 생성된 EventBridge 대상 정보
output "eventbridge_targets" {
  description = "생성된 EventBridge 대상들"
  value = {
    ec2_recovery_target = length(aws_cloudwatch_event_target.ec2_recovery_target) > 0 ? {
      rule      = aws_cloudwatch_event_target.ec2_recovery_target[0].rule
      target_id = aws_cloudwatch_event_target.ec2_recovery_target[0].target_id
      arn       = aws_cloudwatch_event_target.ec2_recovery_target[0].arn
    } : null
    
    s3_cleanup_target = length(aws_cloudwatch_event_target.s3_cleanup_target) > 0 ? {
      rule      = aws_cloudwatch_event_target.s3_cleanup_target[0].rule
      target_id = aws_cloudwatch_event_target.s3_cleanup_target[0].target_id
      arn       = aws_cloudwatch_event_target.s3_cleanup_target[0].arn
    } : null
  }
}