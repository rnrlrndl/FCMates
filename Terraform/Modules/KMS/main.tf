module "kms" {
  source  = "terraform-aws-modules/kms/aws"
  version = "~> 4.0"

  aliases     = ["ex-sns"]
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

  tags = {
    name = "ex-sns"
  }
}