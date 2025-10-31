locals {
    name = "FCMates-EC2-Instance"

    tags = {
        Terraform     = "true"
        Environment   = "dev"
        BackupEnabled = "true"  # AWS Backup에서 사용할 태그
        Project       = "FCMates"
        Component     = "Compute"
    }
}

module "ec2_instance" {
    source  = "terraform-aws-modules/ec2-instance/aws"

    # 기본 정보
    name = local.name
    ami           = var.ami_id
    instance_type = "t3.micro"
    key_name      = "FCMates"
    monitoring    = false    # CloudWatch 모니터링 활성화 서버 안정성을 위하면 true로 설정 가능 (true로 설정 시 비용 발생)
    subnet_id     = var.subnet_id

    # 인스턴스 생성 옵션
    # 기본 EC2 인스턴스 생성
    create = true
    # 스팟 인스턴스 생성 비활성화
    create_spot_instance = false
    # EC2의 IAM 역할 생성 활성화
    create_iam_instance_profile = true

    # IAM 역할 설정
    iam_role_description        = "IAM role for EC2 instance of FCMates"
    iam_role_policies = {
        S3FullAccess        = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
        BackupOperatorAccess = "arn:aws:iam::aws:policy/AWSBackupOperatorAccess"
    }
    # 보안 그룹 생성 활성화
    create_security_group = true
    # EIP 생성 활성화
    create_eip = true

    # 루트 볼륨 설정
    root_block_device = {
        encrypted  = true
        type       = "gp3"
    }

    tags = local.tags
}