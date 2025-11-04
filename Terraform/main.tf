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
# 여기서 본인 이메일 주소를 설정하세요!
    notification_email = "rndl28@naver.com"  #실제 이메일로 변경
}

# Backup
module "backup" {
    source = "./Modules/Backup"

    # 다른 모듈들이 생성된 후에 백업 설정
    depends_on = [module.ec2, module.s3, module.kms, module.autoscaling]

    # KMS 키 설정
    kms_key_arn = module.kms.key_arn

    # 백업 대상 리소스 ARN 목록
    backup_resources = [
        module.ec2.ec2_instance_arn,  # EC2 인스턴스
        module.s3.s3_bucket_arn       # S3 버킷
        # Auto Scaling Group의 인스턴스들은 태그 기반으로 자동 선택됨
    ]

    # 백업 제외 리소스 (필요시)
    not_resources = []

    # 태그 기반 선택 조건 (BackupEnabled=true 태그가 있는 리소스만 백업)
    selection_conditions = {
        string_equals = [
            {
                key   = "aws:ResourceTag/BackupEnabled"
                value = "true"
            }
        ]
        string_like      = []
        string_not_equals = []
        string_not_like   = []
    }

    # 백업 스케줄 설정 (한국 시간 고려)
    daily_backup_schedule  = "cron(0 17 ? * * *)"   # 매일 새벽 2시 (UTC+9 기준)
    weekly_backup_schedule = "cron(0 18 ? * SUN *)" # 매주 일요일 새벽 3시 (UTC+9 기준)

    # 보존 정책
    daily_cold_storage_after  = 30   # 30일 후 콜드 스토리지
    daily_delete_after        = 150   # 90일 후 삭제
    weekly_cold_storage_after = 90   # 90일 후 콜드 스토리지  
    weekly_delete_after       = 365  # 1년 후 삭제

    # 태그 설정
    tags = {
        Environment = "dev"
        Project     = "FCMates"
        Component   = "Backup"
    }
}

# Auto Scaling
module "autoscaling" {
    source = "./Modules/AutoScaling"

    # VPC와 EC2가 생성된 후에 Auto Scaling 설정
    depends_on = [module.vpc, module.ec2]

    # VPC 설정
    vpc_id     = module.vpc.vpc_id
    subnet_ids = module.vpc.private_subnets  # Private 서브넷에 인스턴스 생성

    # Auto Scaling 설정
    name_prefix      = "fcmates-asg"
    min_size         = 1
    max_size         = 3
    desired_capacity = 2

    # 인스턴스 설정
    instance_type    = "t3.micro"
    ami_id          = data.aws_ami.ubuntu.id  # Ubuntu AMI 사용
    key_name        = "FCMates"

    # 스케일링 정책
    target_cpu_utilization = 70

    # 보안 설정
    ssh_cidr_blocks = module.vpc.vpc_cidr_block

    # 알림 설정
    enable_notifications = true
    sns_topic_arn       = module.sns.sns_topic_arn

    # AWS 리전
    aws_region = "ap-northeast-2"

    # 태그 설정
    tags = {
        Environment = "dev"
        Project     = "FCMates"
        Component   = "AutoScaling"
    }
}

# Lambda (복구 자동화)
module "lambda" {
    source = "./Modules/Lambda"

    # 다른 모듈들이 생성된 후에 Lambda 설정
    depends_on = [module.ec2, module.s3, module.cloudwatch, module.sns, module.autoscaling]

    # 기본 설정
    name_prefix = "fcmates"
    aws_region  = "ap-northeast-2"

    # 연동할 서비스 정보
    ec2_instance_id = module.ec2.ec2_instance_id
    s3_bucket_name  = "fcmates-bucket"
    sns_topic_arn   = module.sns.sns_topic_arn

    # Lambda 함수들 정의
    lambda_functions = {
        "ec2-recovery" = {
            source_code = file("${path.module}/Modules/Lambda/ec2_recovery.py")
            timeout     = 300
            environment_variables = {
                CPU_THRESHOLD    = "90"
                ENABLE_SNAPSHOT  = "true"
            }
        }
        
        "s3-cleanup" = {
            source_code = file("${path.module}/Modules/Lambda/s3_cleanup.py")
            timeout     = 600
            environment_variables = {
                S3_CLEANUP_DAYS = "30"
                DRY_RUN        = "false"
            }
        }
        
        "cloudwatch-response" = {
            source_code = file("${path.module}/Modules/Lambda/cloudwatch_response.py")
            timeout     = 180
            environment_variables = {
                AUTO_SCALING_ENABLED = "true"
                MAX_SCALE_OUT        = "3"
            }
        }
    }

    # 스케줄 설정
    s3_cleanup_schedule = "cron(0 6 * * ? *)"  # 매일 오전 6시 (UTC) = 한국 시간 15시

    # 로그 설정
    log_retention_days = 14

    # 복구 설정
    enable_ec2_auto_recovery = true
    enable_s3_cleanup       = true
    s3_cleanup_days         = 30

    # 태그 설정
    tags = {
        Environment = "dev"
        Project     = "FCMates"
        Component   = "Lambda"
    }

}
