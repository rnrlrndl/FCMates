# 현재 리전에 가용 영역이 있는지 AWS에서 조회
# data는 provider를 통해 데이터 불러오기 용도로 사용 
data "aws_availability_zones" "available" {}

# locals는 현재 Terraform 코드 내에서 사용할 지역 변수 설정하는 용도로 사용 
locals {
    # vpc 이름 설정    
    name   = "my-vpc"
    # 리전 설정
    region = "ap-northeast-2"
    # VPC의 기본 CIDR 블록 (사설 네트워크 대역 설정)
    vpc_cidr = "10.0.0.0/16"
    # 가용 영역(Availability Zone) 3개를 자동으로 조회해 리스트로 저장
    azs      = slice(data.aws_availability_zones.available.names, 0, 3)

    # 태그 설정
    tags = {
        Name    = local.name
        Terraform = "true"
        Environment = "dev"
    }
}

# EIP 제한으로 인해 주석 처리 - VPC 모듈이 자동으로 생성
# resource "aws_eip" "nat" {
#     # NAT Gateway를 1개만 사용하여 EIP 절약
#     count = 1

#     tags = merge(local.tags, {
#     Name = "nat-eip-${count.index + 1}"
#     })
# }

module "vpc" {
    source = "terraform-aws-modules/vpc/aws"

    name = local.name
    cidr = local.vpc_cidr

    # AZ 설정 
    azs = local.azs 

    # 각 AZ당 서브넷 1개씩 생성
    public_subnets = [
        "10.0.1.0/24",  
        "10.0.2.0/24",  
        "10.0.3.0/24",  
    ]

    private_subnets = [
        "10.0.11.0/24", 
        "10.0.12.0/24",
        "10.0.13.0/24", 
    ]  

    # DNS 설정
    enable_dns_hostnames = true
    enable_dns_support   = true

    # NAT Gateway 설정
    # VPC 새로 생성 시 새 IP 할당되고, VPC 파괴 시 해당 IP 해제
    # 따라서 VPC 재생성시 동일한 IP 유지하는 것이 편리
    enable_nat_gateway = true
    single_nat_gateway  = true  # 단일 NAT Gateway 사용 (비용 절약)
    one_nat_gateway_per_az = false
    reuse_nat_ips       = false  # EIP 제한으로 인해 자동 생성 사용
    # external_nat_ip_ids = aws_eip.nat[*].id   # EIP 제한으로 인해 주석 처리

    # VPN Gateway 필요 없으면 false로 설정 가능
    enable_vpn_gateway = false

    tags = local.tags
}