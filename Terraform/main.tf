data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical (Ubuntu 공식 계정)

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# VPC
module "vpc" {
  source = "./Modules/VPC"
}

# EC2
module "ec2" {
    source = "./Modules/EC2"

    # VPC가 먼저 생성되어야 하므로 의존성 명시
    depends_on = [module.vpc]

    subnet_id = module.vpc.public_subnets[0]

    ami_id     = data.aws_ami.ubuntu.id
}

# S3
module "s3" {  
  source = "./Modules/S3"

  # EC2에서 IAM 역할로 S3 접근하므로 EC2 이후에 생성
  depends_on = [module.ec2]
}

# CloudWatch
module "cloudwatch" {
    source = "./Modules/CloudWatch"

    # EC2와 S3가 생성된 후에 CloudWatch 설정
    depends_on = [module.ec2, module.s3]

    # EC2 모니터링 설정
    ec2_instance_id   = module.ec2.ec2_instance_id
    ec2_instance_name = "FCMates-EC2-Instance"

    # S3 모니터링 설정
    s3_bucket_name = "fcmates-bucket"
    enable_s3_alarm = true

    # 알람 임계값 설정
    cpu_threshold    = 80
    memory_threshold = 80
    disk_threshold   = 85

    # 메모리 및 디스크 알람 비활성화 (CloudWatch Agent 설치 필요)
    enable_memory_alarm = false
    enable_disk_alarm   = false

    # 대시보드 및 로그 그룹 생성
    create_dashboard = true
    create_log_group = true
    log_retention_days = 7

    # AWS 리전
    aws_region = "ap-northeast-2"

    # SNS Topic 연결
    sns_topic_arn = module.sns.sns_topic_arn
}

# KMS
module "kms" {
    source = "./Modules/KMS"
}

# SNS
module "sns" {
    source = "./Modules/SNS"

    kms_key_id = module.kms.key_id

    # 태그나 리전은 상위 locals를 활용하거나 직접 명시
    providers = {
        aws = aws
    }
}