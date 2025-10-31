import json
import boto3
import os
from datetime import datetime, timedelta

def lambda_handler(event, context):
    """
    S3 버킷 자동 정리 Lambda 함수
    오래된 파일들을 삭제하고 버킷을 최적화합니다.
    """
    
    # AWS 클라이언트 초기화
    s3 = boto3.client('s3', region_name=os.environ.get('AWS_REGION', 'ap-northeast-2'))
    sns = boto3.client('sns', region_name=os.environ.get('AWS_REGION', 'ap-northeast-2'))
    
    # 환경 변수에서 설정 값 가져오기
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    sns_topic_arn = os.environ.get('SNS_TOPIC_ARN')
    cleanup_days = int(os.environ.get('S3_CLEANUP_DAYS', '30'))
    
    try:
        print(f"S3 정리 프로세스 시작: 버킷 {bucket_name}")
        
        # 정리 기준 날짜 계산 (현재 시간 - cleanup_days)
        cutoff_date = datetime.utcnow() - timedelta(days=cleanup_days)
        print(f"정리 기준 날짜: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')} UTC 이전 파일들")
        
        cleanup_stats = {
            'total_objects_checked': 0,
            'objects_deleted': 0,
            'bytes_freed': 0,
            'deleted_objects': [],
            'multipart_uploads_aborted': 0
        }
        
        # 1. 일반 객체 정리
        paginator = s3.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket_name):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                cleanup_stats['total_objects_checked'] += 1
                
                # 객체의 마지막 수정 시간이 기준 날짜보다 오래된 경우
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    try:
                        # 객체 삭제
                        s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
                        
                        cleanup_stats['objects_deleted'] += 1
                        cleanup_stats['bytes_freed'] += obj['Size']
                        cleanup_stats['deleted_objects'].append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat()
                        })
                        
                        print(f"삭제됨: {obj['Key']} ({obj['Size']} bytes)")
                        
                    except Exception as e:
                        print(f"객체 삭제 실패 {obj['Key']}: {str(e)}")
        
        # 2. 완료되지 않은 멀티파트 업로드 정리
        try:
            multipart_uploads = s3.list_multipart_uploads(Bucket=bucket_name)
            
            if 'Uploads' in multipart_uploads:
                for upload in multipart_uploads['Uploads']:
                    # 시작된 지 1일 이상 된 멀티파트 업로드 중단
                    if upload['Initiated'].replace(tzinfo=None) < datetime.utcnow() - timedelta(days=1):
                        s3.abort_multipart_upload(
                            Bucket=bucket_name,
                            Key=upload['Key'],
                            UploadId=upload['UploadId']
                        )
                        cleanup_stats['multipart_uploads_aborted'] += 1
                        print(f"멀티파트 업로드 중단: {upload['Key']}")
                        
        except Exception as e:
            print(f"멀티파트 업로드 정리 중 오류: {str(e)}")
        
        # 3. 버킷 메트릭 수집
        try:
            # 현재 버킷 크기 확인 (CloudWatch에서)
            cloudwatch = boto3.client('cloudwatch', region_name=os.environ.get('AWS_REGION', 'ap-northeast-2'))
            
            bucket_size_response = cloudwatch.get_metric_statistics(
                Namespace='AWS/S3',
                MetricName='BucketSizeBytes',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': bucket_name},
                    {'Name': 'StorageType', 'Value': 'StandardStorage'}
                ],
                StartTime=datetime.utcnow() - timedelta(hours=24),
                EndTime=datetime.utcnow(),
                Period=86400,  # 24시간
                Statistics=['Average']
            )
            
            current_bucket_size = 0
            if bucket_size_response['Datapoints']:
                current_bucket_size = bucket_size_response['Datapoints'][-1]['Average']
                
        except Exception as e:
            print(f"버킷 메트릭 수집 중 오류: {str(e)}")
            current_bucket_size = 0
        
        # 4. 정리 결과 요약
        freed_mb = cleanup_stats['bytes_freed'] / (1024 * 1024)
        
        summary_message = {
            "timestamp": datetime.utcnow().isoformat(),
            "bucket_name": bucket_name,
            "cleanup_criteria": f"Files older than {cleanup_days} days",
            "cutoff_date": cutoff_date.isoformat(),
            "statistics": {
                "total_objects_checked": cleanup_stats['total_objects_checked'],
                "objects_deleted": cleanup_stats['objects_deleted'],
                "bytes_freed": cleanup_stats['bytes_freed'],
                "mb_freed": round(freed_mb, 2),
                "multipart_uploads_aborted": cleanup_stats['multipart_uploads_aborted'],
                "current_bucket_size_bytes": current_bucket_size
            },
            "deleted_objects_sample": cleanup_stats['deleted_objects'][:10]  # 처음 10개만 표시
        }
        
        print(f"S3 정리 완료: {cleanup_stats['objects_deleted']}개 객체, {freed_mb:.2f}MB 정리됨")
        
        # 5. SNS 알림 전송
        sns.publish(
            TopicArn=sns_topic_arn,
            Subject=f"FCMates S3 정리 완료 - {bucket_name}",
            Message=json.dumps(summary_message, indent=2, ensure_ascii=False)
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'S3 cleanup completed successfully',
                'bucket_name': bucket_name,
                'objects_deleted': cleanup_stats['objects_deleted'],
                'bytes_freed': cleanup_stats['bytes_freed']
            })
        }
        
    except Exception as e:
        error_message = f"S3 정리 중 오류 발생: {str(e)}"
        print(error_message)
        
        # 오류 알림 전송
        if sns_topic_arn:
            sns.publish(
                TopicArn=sns_topic_arn,
                Subject=f"FCMates S3 정리 실패 - {bucket_name}",
                Message=json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "bucket_name": bucket_name,
                    "error": error_message,
                    "event": event
                }, indent=2, ensure_ascii=False)
            )
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': error_message,
                'bucket_name': bucket_name
            })
        }