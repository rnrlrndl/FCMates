data "aws_caller_identity" "current" {}

resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

locals {
    region = "ap-northeast-2"
    name   = "ex-${basename(path.cwd)}"

    tags = {
        Name       = local.name
    }
}

module "sns_topic" {
    source = "terraform-aws-modules/sns/aws"

    name              = local.name
    # 같은 이름 중복 방지
    use_name_prefix   = true
    # 콘솔에 보이는 이름
    display_name      = "FCMates-Notification"
    # SNS 메시지 암호용 KMS 키
    kms_master_key_id = var.kms_key_id
    # X-Ray 트레이싱 활성화
    tracing_config    = "Active"

    # 비활성화 (실시간 알림용 구조)
    # FIFO로 토픽 생성 
    fifo_topic                  = false
    # 내용이 같으면 중복 메시지 제거
    content_based_deduplication = false

    create_topic_policy         = true      # Terraform이 SNS 토픽용 IAM 정책을 자동 생성하도록 설정
    enable_default_topic_policy = true      # AWS가 제공하는 기본 토픽 정책을 함께 병합하도록 설정

    topic_policy_statements = {
        # Publish 권한 - 같은 AWS 계정 내 서비스만 허용
        pub = {
            actions = ["sns:Publish"]
            principals = [{
                type        = "AWS"
                identifiers = [data.aws_caller_identity.current.arn]
            }]
        }

        # Subscribe 권한
        sub = {
            actions = ["sns:Subscribe", "sns:Receive"]
            principals = [{
                type        = "Service"
                identifiers = ["lambda.amazonaws.com"]
            }]
        }
    }
# 이메일 구독 설정 - 하나의 이메일만 사용
    subscriptions = {
        email_notification = {
            protocol = "email"
            endpoint = var.notification_email
        }
    }

    # SNS가 Lambda 호출 결과를 CloudWatch에 기록
    lambda_feedback = {
        failure_role_arn    = aws_iam_role.sns_feedback.arn
        success_role_arn    = aws_iam_role.sns_feedback.arn
        success_sample_rate = 100
    }

    tags = local.tags
}

resource "aws_iam_role" "sns_feedback" {
    name = "fcmates-sns-feedback-role-${random_string.suffix.result}"

    assume_role_policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
            {
                Sid    = "AllowSNSAssumeRole"
                Effect = "Allow"
                Action = "sts:AssumeRole"
                Principal = {
                    Service = "sns.amazonaws.com"
                }
            }
        ]
    })

    tags = local.tags
}

resource "aws_iam_role_policy" "sns_feedback_policy" {
    name = "${local.name}-sns-feedback-policy"
    role = aws_iam_role.sns_feedback.id

    policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
            {
                Sid    = "AllowLogs"
                Effect = "Allow"
                Action = [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ]
                Resource = "*"
            }
        ]
    })

}
