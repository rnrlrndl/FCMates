# SNS 토픽 

output "sns_topic_arn" {
  description = "The ARN of the SNS topic"
  value       = module.sns_topic.topic_arn
}

output "sns_topic_name" {
  description = "The name of the SNS topic"
  value       = module.sns_topic.topic_name
}

output "sns_topic_id" {
  description = "The ID of the SNS topic (same as ARN)"
  value       = module.sns_topic.topic_id
}

output "sns_topic_owner" {
  description = "The AWS Account ID of the SNS topic owner"
  value       = module.sns_topic.topic_owner
}

output "sns_subscriptions" {
  description = "Map of subscriptions created and their attributes"
  value       = module.sns_topic.subscriptions
}

# 람다 피드백용 IAM 역할 및 정책 출력

output "sns_feedback_role_arn" {
  description = "IAM Role ARN used by SNS for feedback logging"
  value       = aws_iam_role.sns_feedback.arn
}

output "sns_feedback_policy_name" {
  description = "The name of the IAM policy attached to the SNS feedback role"
  value       = aws_iam_role_policy.sns_feedback_policy.name
}
