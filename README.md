# 💻 클라우드 기반 자동 백업 및 서비스               복구 플랫폼

👋 FCMATE 관련 주제 선정 및 필요성 문서

📌 제안 목적

- 실제 서비스에서 장애가 나면 동작할 수 없는 문제가 발생

        →서버가 고장 나도 자동으로 복구되는 시스템 설계

- 백업과 복원 시스템을 직접 구성해보는 클라우드 실습
- 복잡한 코딩 없이도 실무에 가까운 경험 가능

## 🧠 어려운 단어 설명

| 용어 | 뜻 |
| --- | --- |
| **EC2** | AWS에서 제공하는 가상 서버 (우리가 웹사이트 올릴 공간) |
| **S3** | 파일 저장소 (백업용 하드디스크 같은 개념) |
| **Auto Scaling** | 서버가 터지면 자동으로 새 서버를 만들어주는 기능 |
| **CloudWatch** | 서버 상태를 감시하는 도구 (CPU, 메모리 확인 가능) |
| **SNS** | 알림 기능. 서버 문제 생기면 문자/이메일로 알려줘 |
| **IAM/VPC** | 보안 기능 (누가 접속 가능한지, 어디서 접속 가능한지 설정함) |

### 🧠 우리가 배울 기술 (AWS)

- EC2: 웹 서버 띄우기
- S3: 데이터 백업
- Auto Scaling: 서버 죽으면 자동 복구
- CloudWatch: 상태 모니터링
- SNS: 알림 보내기
- IAM, VPC: 보안 설정

### 🎯 프로젝트 목표

- 간단한 웹사이트를 EC2에 띄우고
- 중요한 데이터를 S3로 자동 백업하고
- 장애 발생 시 Auto Scaling으로 복구되고
- 알림도 자동으로 받는 시스템을 구축!

### 🧰 실습 준비물

- AWS 계정 (무료)
- HTML 템플릿
- 기본 리눅스 명령어 (cd, ls, vi 등)
- 팀워크 + 문서정리 도구 (Notion, Google Docs 등)

### 

## ✅ 대체 가능한 GitHub 자료 & 예제들

| 리포지토리 / 글 | 내용 요약 | 우리 프로젝트와 유사한 점 |
| --- | --- | --- |
| **vivaldosantanawebdev/p2-aws-automated-backup** [GitHub](https://github.com/vivaldosantanawebdev/p2-aws-automated-backup?utm_source=chatgpt.com) | EC2에서 IAM Role + cron job 써서 S3로 파일 자동 백업하는 예제. shell 스크립트 포함됨. [GitHub](https://github.com/vivaldosantanawebdev/p2-aws-automated-backup?utm_source=chatgpt.com) | 백업 자동화 + IAM 역할 설정 어떻게 하는지 배우기 좋음. 우리 “EC2 → S3 백업” 파트 참고용 |
| **MrinmoiHossain/Deploy-a-Static-Website-on-AWS** [GitHub](https://github.com/MrinmoiHossain/Deploy-a-Static-Website-on-AWS?utm_source=chatgpt.com) | HTML/CSS/JS 정적 사이트를 S3 버킷 + IAM 보안정책 + CloudFront 배포 예시 있음. [GitHub](https://github.com/MrinmoiHossain/Deploy-a-Static-Website-on-AWS?utm_source=chatgpt.com) | 정적 웹페이지 배포 + CloudFront 설정 참고 가능. “정적 웹사이트 + 보안 정책” 연습에 유리함 |
| **aws-samples/backup-recovery-with-aws-backup** [GitHub](https://github.com/aws-samples/backup-recovery-with-aws-backup?utm_source=chatgpt.com) | 여러 리전/계정에서 AWS Backup 활용하여 백업/복구 솔루션 구성 예시. [GitHub](https://github.com/aws-samples/backup-recovery-with-aws-backup?utm_source=chatgpt.com) | 백업/복구 아키텍처 이해 + 복구 전략 설계할 때 유용 |
| **aws-samples/authenticated-static-site** [GitHub](https://github.com/aws-samples/authenticated-static-site?utm_source=chatgpt.com) | 비공개 S3 + CloudFront + 사용자 인증 (Cognito / Lambda@Edge) 설정 예시. [GitHub](https://github.com/aws-samples/authenticated-static-site?utm_source=chatgpt.com) | 보안 + 배포 + 정적 호스팅 조합. 심화 옵션으로 팀 수준 높이는 데 좋음 |

---

## 💡 우리 팀에 유용하게 쓸 수 있는 자료 + 활용 방식

- “Automated Backup” 예제 보고, 우리 백업 스크립트 & cron job 짜기
- “Static Website 배포 + CloudFront” 예제 보고, S3에 정적 사이트 넣고 공개 설정 + 캐시(CCD) 실습
- “Backup‑Recovery” 샘플 보고, 장애 복구 시나리오 설계 도움
- “Authenticated Static Site” 보고 보안 설정 배워서 적용하면 교수님한테 어필 가능

### 📈 예상 결과물

- Auto Healing 시연 영상
- 백업 성공 로그 캡처
- 시스템 구성도
- 발표 PPT + 실습 매뉴얼

간단 요약문

-1. 서버 한대를 만든다. 

-2. 그 서버가 꺼졌을 때 자동으로 복구 되게 만든다.

-3. 서버 안의 중요한 데이터를 자동으로 저장 한다(S3방식으로)

-4. 관련 문제가 생길 시 문자 또는 이메일로 알림을 보낸다.

-5. 위의 과정을 담은 흐롬을 구성도와 문서로 정리

-6. 위사항들을 발표.

## 🎯 최종 상태 예시

- 브라우저에서 `우리 홈페이지`에 접속 가능
- 서버 꺼뜨려도 몇 분 후 다시 살아남
- EC2 서버 안의 파일은 S3에 자동 저장됨
- 문제가 생기면 우리 메일로 알림 옴
- 이 과정을 문서, PPT, 영상으로 정리함

## 🔧 결국 요약하면?

> “서버 만들고, 죽어도 다시 살아나고,
> 
> 
> 자료도 지켜지고,
> 
> 문제가 생기면 알려주는
> 
> 자동 시스템을 우리가 직접 만든다.”
>
