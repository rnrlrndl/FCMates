data "aws_caller_identity" "current" {}

resource "random_pet" "this" {
  length = 2
}

resource "aws_kms_key" "objects" {
  description             = "KMS key is used to encrypt bucket objects"
  deletion_window_in_days = 7
}

module "log_bucket" {
    source = "terraform-aws-modules/s3-bucket/aws"

    bucket = "logs-${random_pet.this.id}"   
    force_destroy = true

    # 객체 소유권 설정 (버킷 소유자 기준)
    control_object_ownership = true
    object_ownership         = "BucketOwnerPreferred"

    # 보안 설정 (HTTP 차단 + TLS 강제)
    attach_deny_insecure_transport_policy = true
    attach_require_latest_tls_policy      = true

    # 접근 로그 수집을 허용 (다른 S3 버킷이 로그를 이 버킷으로 전송할 수 있게 함)
    attach_access_log_delivery_policy = true

    # 로그 접근 정책: 내 AWS 계정만 접근 가능
    access_log_delivery_policy_source_accounts = [data.aws_caller_identity.current.account_id]

    # CloudTrail, WAF, LB 로그 정책은 불필요하므로 제거
    # attach_elb_log_delivery_policy        = false
    # attach_lb_log_delivery_policy         = false
    # attach_cloudtrail_log_delivery_policy = false
    # attach_waf_log_delivery_policy        = false

    tags = {
        Name        = "fcmates-log-bucket"
        Environment = "dev"
        Purpose     = "EC2/S3 backup log storage"
    }
}

module "s3_bucket" {
    source = "terraform-aws-modules/s3-bucket/aws"

    bucket = "fcmates-bucket"

    # 접근 권한 설정
    acl    = "private"

    # 버킷 안에 로그가 남아있어도 버킷 삭제 가능 (true로 설정 시)
    force_destroy = true

    # 소유권 설정 
    control_object_ownership = true
    object_ownership         = "BucketOwnerPreferred"

    # HTTP 접근 차단 + TLS 최신버전 강제 + 암호화 강제
    attach_deny_insecure_transport_policy     = true    # HTTPS만 허용
    attach_require_latest_tls_policy          = true    # TLS 1.2 이상만 허용

    # 암호화 일관성 유지 (KMS 암호화 강제)
    attach_deny_unencrypted_object_uploads    = true    # 암호화되지 않은 업로드 차단
    attach_deny_incorrect_kms_key_sse         = true    # 지정된 KMS 키 외 사용 금지
    allowed_kms_key_arn                       = aws_kms_key.objects.arn  # 허용된 KMS 키

    # SSE-C(사용자 제공 키) 암호화 거부
    attach_deny_ssec_encrypted_object_uploads = true    # 로컬키 기반 암호화 거부

    # 버킷 소유자 외 접근 차단 
    expected_bucket_owner = data.aws_caller_identity.current.account_id

    # 라이프사이클 규칙 설정
    transition_default_minimum_object_size = "varies_by_storage_class"

    # EC2 -> S3 백업 시 로그 생성
    logging = {
        target_bucket = module.log_bucket.s3_bucket_id
        target_prefix = "access-logs/"
        target_object_key_format = {
            partitioned_prefix = {
                partition_date_source = "DeliveryTime" # "EventTime"
            }
        }
    }

    # 버전 관리 비활성화
    versioning = {
        status     = false # true로 설정 시 기존 파일 보존 하지만 비용 발생
        mfa_delete = false
    }

    # EC2나 AWS Backup이 데이터를 업로드할 때 자동으로 KMS 기반 암호화 적용
    server_side_encryption_configuration = {
        rule = {
                apply_server_side_encryption_by_default = {
                    kms_master_key_id = aws_kms_key.objects.arn
                    sse_algorithm     = "aws:kms"
                }
            }
    }

    # 라이프사이클 설정
    lifecycle_rule = [
        {
            id      = "backup-lifecycle"
            enabled = true
            transition = [
                {
                    days          = 30
                    storage_class = "STANDARD_IA"
                },
                {
                    days          = 60
                    storage_class = "GLACIER"
                }
            ]
            expiration = {
                days = 90
            }
        }
    ]

    # 접근 빈도에 따른 저장소 이동 (비용 절감)
    intelligent_tiering = {
        backup = {
            status = "Enabled"
            filter = {
                prefix = "/"
            }
            tiering = {
                ARCHIVE_ACCESS = { days = 90 }
                DEEP_ARCHIVE_ACCESS = { days = 180 }
            }
        }
    }

    # CloudWatch 메트릭 생성
    metric_configuration = [
        {
            name = "backup"
            filter = {
                prefix = "backup/"
                tags = {
                    Environment = "dev"
                }
            }
        },
        {
            name = "critical-logs"
            filter = {
                prefix = "logs/"
                tags = {
                    Importance = "high"
                }
            }
        },
        {
            name = "all"
        }
    ]

    tags = {
        Owner         = "FCMates"
        Environment   = "dev"
        BackupEnabled = "true"  # AWS Backup에서 사용할 태그
        Project       = "FCMates"
        Component     = "Storage"
    }
}