data "aws_caller_identity" "current" {}

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
    kms_master_key_id = module.kms.key_id
    # X-Ray 트레이싱 활성화
    tracing_config    = "Active"

    # 비활성화 (실시간 알림용 구조)
    # FIFO로 토픽 생성 
    fifo_topic                  = false
    # 처리량 단위 (토픽 or 그룹)
    fifo_throughput_scope       = "MessageGroup"
    # 내용이 같으면 중복 메시지 제거
    content_based_deduplication = false

    delivery_policy = jsonencode({
        "lambda": {
            "defaultHealthyRetryPolicy": {
                "minDelayTarget": 5,                # 5초 후 재시도 시작
                "maxDelayTarget": 60,               # 최대 1분 간격까지 증가
                "numRetries": 5,                    # 최대 5회 재시도
                "backoffFunction": "exponential",   # 재시도 간격을 지수적으로 증가시킴
                "numNoDelayRetries": 1,             # 첫 번째는 즉시 재시도
                "numMinDelayRetries": 2,            # 이후 5초 간격 2회
                "numMaxDelayRetries": 2             # 마지막엔 60초 간격으로 2회
            },
            "disableSubscriptionOverrides": true,   # 커스터마이징 방지
            "defaultThrottlePolicy": {
                "maxReceivesPerSecond": 5           # 초당 최대 5회 처리 허용
            }
        }
    })

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

    # SNS → Lambda 직접 연결
    subscriptions = {
        lambda = {
            protocol = "lambda"
            endpoint = aws_lambda_function.sns_handler.arn
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

module "kms" {
  source  = "terraform-aws-modules/kms/aws"
  version = "~> 4.0"

  aliases     = ["sns/${local.name}"]
  description = "KMS key to encrypt topic"

  # Policy
  key_statements = [
    {
      sid = "SNS"
      actions = [
        "kms:GenerateDataKey*",
        "kms:Decrypt"
      ]
      resources = ["*"]
      principals = [{
        type        = "Service"
        identifiers = ["sns.amazonaws.com"]
      }]
    }
  ]

  tags = local.tags
}

resource "aws_iam_role" "sns_feedback" {
    name = "${local.name}-sns-feedback-role"

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