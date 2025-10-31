module "kms" {
  source  = "terraform-aws-modules/kms/aws"
  version = "~> 4.0"

  # aliases 제거하여 충돌 방지 - Key ARN으로 직접 참조
  description = "KMS key to encrypt FCMates SNS topic"

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

  tags = {
    name = "fcmates-sns"
    Environment = "dev"
    Project = "FCMates"
  }
}