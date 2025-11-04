# FCMates AWS 자동화 파이프라인

**프로젝트**: FCMates  
**작성일**: 2025년 11월 4일  
**목적**: AWS 기반 완전 자동화 인프라 운영 시스템

---

## 🔄 **전체 아키텍처 개요**

```
CloudWatch Metrics → Lambda Functions → AWS Services
                ↓
            SNS Notifications
                ↓
         관리자/운영팀 알림
```

### **핵심 구성 요소**
- **AWS Backup 자동화**: 정기 백업 및 복구 시스템
- **SNS 알림 시스템**: 실시간 상태 알림 및 경보
- **Terraform 기반 인프라**: 코드로 관리되는 클라우드 인프라
- **Auto Scaling 구성**: 트래픽 기반 자동 확장
- **CloudWatch 모니터링**: 모든 작업의 상태 점검

---

## 📊 **파이프라인 상세 흐름**

### **Pipeline 1: 트래픽 기반 Auto Scaling**
```
1. CloudWatch → CPU/Network 메트릭 수집 (5분 간격)
2. 임계값 초과 감지 → CloudWatch Alarm 발생
3. EventBridge → Lambda 함수 트리거
4. Lambda → Auto Scaling Group 확장 명령
5. SNS → 스케일링 알림 발송
6. CloudWatch → 새 인스턴스 상태 모니터링
```

**📋 세부 동작:**
- **트리거 조건**: CPU > 70% (5분간 지속)
- **스케일링 범위**: 현재 1개 → 최대 3개 인스턴스
- **알림 내용**: "Auto Scaling 활성화 - 인스턴스 2개 추가 생성"
- **쿨다운 시간**: 300초 (중복 스케일링 방지)

### **Pipeline 2: 장애 감지 및 백업 복구**
```
1. CloudWatch → 인스턴스 상태 체크 (1분 간격)
2. 장애 감지 → StatusCheckFailed Alarm
3. EventBridge → Lambda 복구 함수 실행
4. Lambda → 최신 백업 스냅샷 조회
5. Lambda → 백업본으로 인스턴스 복구
6. SNS → 복구 완료 알림
7. CloudWatch → 복구된 인스턴스 모니터링
```

**📋 세부 동작:**
- **트리거 조건**: Status Check Failed (2회 연속)
- **복구 방식**: 최신 스냅샷으로 새 인스턴스 생성
- **복구 시간**: 평균 5분 이내
- **알림 내용**: "인스턴스 장애 감지 → 백업 복구 완료"
- **롤백 옵션**: 복구 실패 시 이전 백업본 사용

### **Pipeline 3: 정기 백업 자동화**
```
1. EventBridge → 스케줄 기반 Lambda 실행 (매일 새벽 2시)
2. Lambda → EC2/S3 백업 생성
3. AWS Backup → 스냅샷 및 버킷 백업 수행
4. Lambda → 백업 상태 확인
5. CloudWatch → 백업 성공/실패 메트릭 기록
6. SNS → 백업 결과 알림
```

**📋 세부 동작:**
- **백업 주기**: 
  - 일일: 매일 새벽 2시 (UTC+9)
  - 주간: 매주 일요일 새벽 3시
- **보존 정책**: 
  - 일일 백업: 90일 후 삭제
  - 주간 백업: 1년 후 삭제
- **백업 대상**: EC2 인스턴스, S3 버킷
- **알림 내용**: "백업 완료 - 12개 스냅샷 생성"
- **암호화**: KMS 키를 통한 백업 데이터 암호화

### **Pipeline 4: S3 스토리지 정리**
```
1. EventBridge → 스케줄 Lambda 실행 (매일 오후 3시)
2. Lambda → S3 객체 스캔 (30일 이상 파일)
3. Lambda → 불필요한 파일 삭제
4. CloudWatch → 삭제된 파일 수 메트릭
5. SNS → 정리 완료 알림
```

**📋 세부 동작:**
- **정리 주기**: 매일 오후 3시 (한국 시간)
- **삭제 조건**: 30일 이상된 임시 파일
- **예외 처리**: 중요 파일 태그 검사
- **알림 내용**: "S3 정리 완료 - 156MB 용량 확보"

---

## 🛠 **핵심 Lambda 함수별 역할**

### **1. `ec2_recovery.py` - 인스턴스 복구**
```python
def lambda_handler(event, context):
    """
    EC2 인스턴스 자동 복구 Lambda 함수
    장애 발생 시 최신 백업으로 인스턴스 복구
    """
    # 1. 장애 인스턴스 식별
    # 2. 최신 백업 스냅샷 조회
    # 3. 새 인스턴스 생성 및 복구
    # 4. SNS 알림 발송
    # 5. CloudWatch 메트릭 업데이트
```

**주요 기능:**
- 인스턴스 상태 모니터링
- 자동 스냅샷 생성
- 장애 시 즉시 복구
- 복구 로그 CloudWatch 전송

### **2. `cloudwatch_response.py` - Auto Scaling 제어**
```python
def lambda_handler(event, context):
    """
    CloudWatch 메트릭 기반 Auto Scaling 제어
    트래픽 증가 시 자동 인스턴스 확장
    """
    # 1. CloudWatch 알람 이벤트 분석
    # 2. CPU/네트워크 메트릭 확인
    # 3. Auto Scaling Group 조정
    # 4. SNS 스케일링 알림
    # 5. 새 인스턴스 모니터링 시작
```

**주요 기능:**
- 실시간 메트릭 분석
- 동적 스케일링 결정
- 비용 최적화 로직
- 스케일링 이력 추적

### **3. `s3_cleanup.py` - 스토리지 관리**
```python
def lambda_handler(event, context):
    """
    S3 스토리지 자동 정리 시스템
    불필요한 파일 삭제로 비용 절감
    """
    # 1. S3 버킷 객체 스캔
    # 2. 30일 이상 파일 식별
    # 3. 불필요한 파일 삭제
    # 4. 정리 결과 SNS 알림
```

**주요 기능:**
- 정기적 스토리지 점검
- 비용 절감 자동화
- 중요 파일 보호
- 정리 통계 제공

---

## 📱 **SNS 알림 시스템 구성**

### **알림 카테고리별 메시지**
```
🔴 긴급 (CRITICAL):
   "EC2 인스턴스 장애 감지 - 자동 복구 진행 중"
   "백업 복구 실패 - 수동 개입 필요"

🟡 경고 (WARNING):
   "CPU 사용률 80% 초과 - Auto Scaling 활성화"
   "디스크 사용률 85% 도달 - 용량 확장 권장"

🟢 정상 (INFO):
   "일일 백업 완료 - 모든 리소스 정상"
   "Auto Scaling 완료 - 인스턴스 3개로 확장"

🔵 정보 (DEBUG):
   "S3 정리 완료 - 156MB 용량 확보"
   "정기 점검 완료 - 이상 없음"
```

### **알림 대상 및 채널**
```
📧 Email 알림:
├── 개발팀: dev-team@fcmates.com
├── 운영팀: ops-team@fcmates.com
└── 관리자: admin@fcmates.com

📱 SMS 알림:
├── 긴급 상황: +82-10-xxxx-xxxx (관리자)
└── 백업 실패: +82-10-yyyy-yyyy (개발팀장)

💬 Slack 연동:
├── #monitoring: 실시간 모니터링
├── #alerts: 알람 및 경고
└── #backup: 백업 상태 알림
```

### **알림 필터링 규칙**
- **긴급**: 즉시 모든 채널 알림
- **경고**: Email + Slack 알림
- **정상**: Slack 채널만 알림
- **정보**: 로그 기록만 (선택적 알림)

---

## 📈 **CloudWatch 모니터링 체계**

### **핵심 메트릭 대시보드**
```
📊 EC2 지표:
├── CPUUtilization (임계값: 70%)
├── NetworkIn/Out (트래픽 모니터링)
├── StatusCheckFailed (장애 감지)
├── DiskSpaceUtilization (디스크 사용률: 85%)
└── MemoryUtilization (메모리 사용률: 80%)

💾 S3 지표:
├── BucketSizeBytes (스토리지 사용량)
├── NumberOfObjects (객체 수)
├── AllRequests (API 요청 수)
└── 4xxErrors (클라이언트 에러율)

⚡ Lambda 지표:
├── Duration (실행 시간)
├── Errors (에러 발생률)
├── Invocations (호출 횟수)
└── ConcurrentExecutions (동시 실행)

📈 Auto Scaling 지표:
├── GroupDesiredCapacity (원하는 용량)
├── GroupInServiceInstances (서비스 중인 인스턴스)
├── GroupTotalInstances (총 인스턴스)
└── GroupScalingHistory (스케일링 이력)
```

### **대시보드 구성**
```
🖥️ 메인 대시보드:
├── 📊 실시간 모니터링 (CPU, 메모리, 네트워크)
├── 🔄 Auto Scaling 현황 (인스턴스 수, 확장 이력)
├── 💾 백업 상태 (최근 백업 시간, 성공률)
├── ⚠️ 알람 현황 (활성 알람, 해결된 이슈)
└── 💰 비용 추적 (일간/월간 사용 비용)

📱 모바일 대시보드:
├── 🚨 긴급 알람만 표시
├── 📈 핵심 메트릭 요약
└── 🔧 빠른 조치 버튼
```

---

## ⚙️ **Terraform 기반 인프라 구성**

### **모듈 구조**
```
📁 Terraform/
├── 📄 main.tf (메인 설정)
├── 📄 provider.tf (AWS Provider)
├── 📄 variables.tf (변수 정의)
├── 📄 outputs.tf (출력 값)
└── 📁 Modules/
    ├── 📁 VPC/ (네트워크 인프라)
    ├── 📁 EC2/ (컴퓨팅 리소스)
    ├── 📁 S3/ (스토리지 서비스)
    ├── 📁 CloudWatch/ (모니터링)
    ├── 📁 Lambda/ (서버리스 함수)
    ├── 📁 AutoScaling/ (자동 확장)
    ├── 📁 Backup/ (백업 시스템)
    ├── 📁 SNS/ (알림 서비스)
    └── 📁 KMS/ (암호화 관리)
```

### **배포 프로세스**
```
1. terraform init    (초기화)
2. terraform plan    (계획 검토)
3. terraform apply   (배포 실행)
4. terraform destroy (리소스 정리)
```

---

## 🔒 **보안 및 권한 관리**

### **IAM 역할 구성**
```
🔐 Lambda 실행 역할:
├── EC2 관리 권한 (인스턴스 시작/중지/재부팅)
├── S3 접근 권한 (객체 읽기/쓰기/삭제)
├── CloudWatch 권한 (메트릭 조회/알람 관리)
├── SNS 발행 권한 (알림 전송)
└── Auto Scaling 권한 (그룹 용량 조절)

🔐 EC2 인스턴스 역할:
├── S3 읽기 전용 (애플리케이션 데이터)
├── CloudWatch 로그 전송
└── Systems Manager 접근
```

### **보안 정책**
- **최소 권한 원칙**: 필요한 권한만 부여
- **암호화**: 모든 데이터 전송/저장 시 암호화
- **접근 로깅**: 모든 API 호출 CloudTrail 기록
- **네트워크 보안**: VPC, 보안 그룹으로 접근 제한

---

## ⚡ **전체 파이프라인 요약**

```
🔍 모니터링 → ⚠️ 임계값 감지 → 🤖 Lambda 실행 → 🔧 자동 조치 → 📱 SNS 알림 → 📊 결과 추적
```

### **주요 이점**
✅ **무인 운영**: 24/7 자동 모니터링 및 대응  
✅ **빠른 복구**: 장애 발생 시 5분 이내 자동 복구  
✅ **비용 최적화**: 사용량 기반 자동 스케일링  
✅ **데이터 보호**: 정기 자동 백업 및 암호화  
✅ **실시간 알림**: 즉각적인 상황 인지 및 대응  

### **성능 지표**
- **가용성**: 99.9% 이상
- **복구 시간**: 평균 3분
- **비용 절감**: 30% 이상
- **알림 지연**: 1분 이내

---

**마지막 업데이트**: 2025년 11월 4일  
**버전**: v1.0  
**담당자**: FCMates 개발팀