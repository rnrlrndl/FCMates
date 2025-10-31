import json
import boto3
import os
from datetime import datetime

def lambda_handler(event, context):
    """
    EC2 인스턴스 자동 복구 Lambda 함수
    CloudWatch 알람이 ALARM 상태일 때 호출되어 EC2 인스턴스를 복구합니다.
    """
    
    # AWS 클라이언트 초기화
    ec2 = boto3.client('ec2', region_name=os.environ.get('AWS_REGION', 'ap-northeast-2'))
    sns = boto3.client('sns', region_name=os.environ.get('AWS_REGION', 'ap-northeast-2'))
    cloudwatch = boto3.client('cloudwatch', region_name=os.environ.get('AWS_REGION', 'ap-northeast-2'))
    
    # 환경 변수에서 설정 값 가져오기
    instance_id = os.environ.get('EC2_INSTANCE_ID')
    sns_topic_arn = os.environ.get('SNS_TOPIC_ARN')
    
    try:
        print(f"EC2 복구 프로세스 시작: Instance ID {instance_id}")
        
        # 1. 현재 인스턴스 상태 확인
        response = ec2.describe_instances(InstanceIds=[instance_id])
        instance = response['Reservations'][0]['Instances'][0]
        current_state = instance['State']['Name']
        
        print(f"현재 인스턴스 상태: {current_state}")
        
        recovery_actions = []
        
        # 2. 인스턴스 상태에 따른 복구 작업 수행
        if current_state == 'stopped':
            print("인스턴스가 중지된 상태입니다. 시작을 시도합니다.")
            ec2.start_instances(InstanceIds=[instance_id])
            recovery_actions.append("인스턴스 시작")
            
        elif current_state == 'running':
            # 인스턴스가 실행 중이지만 알람이 발생한 경우
            print("인스턴스가 실행 중입니다. 상태를 확인합니다.")
            
            # CPU 사용률 확인
            cpu_stats = cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=datetime.utcnow().replace(minute=0, second=0, microsecond=0),
                EndTime=datetime.utcnow(),
                Period=300,
                Statistics=['Average']
            )
            
            if cpu_stats['Datapoints']:
                latest_cpu = cpu_stats['Datapoints'][-1]['Average']
                print(f"현재 CPU 사용률: {latest_cpu}%")
                
                if latest_cpu > 90:
                    print("높은 CPU 사용률로 인해 인스턴스를 재부팅합니다.")
                    ec2.reboot_instances(InstanceIds=[instance_id])
                    recovery_actions.append("높은 CPU 사용률로 인한 재부팅")
            
        elif current_state in ['stopping', 'pending']:
            print(f"인스턴스가 {current_state} 상태입니다. 잠시 후 다시 시도하세요.")
            recovery_actions.append(f"인스턴스 상태 대기 중: {current_state}")
            
        else:
            print(f"예상치 못한 인스턴스 상태: {current_state}")
            recovery_actions.append(f"알 수 없는 상태: {current_state}")
        
        # 3. 스냅샷 생성 (데이터 보호)
        if current_state in ['running', 'stopped']:
            volumes = [ebs['Ebs']['VolumeId'] for ebs in instance.get('BlockDeviceMappings', [])]
            for volume_id in volumes:
                snapshot_response = ec2.create_snapshot(
                    VolumeId=volume_id,
                    Description=f"Emergency snapshot for {instance_id} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                print(f"스냅샷 생성: {snapshot_response['SnapshotId']}")
                recovery_actions.append(f"볼륨 {volume_id} 스냅샷 생성")
        
        # 4. SNS 알림 전송
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "instance_id": instance_id,
            "previous_state": current_state,
            "recovery_actions": recovery_actions,
            "event_detail": event.get('detail', {})
        }
        
        sns.publish(
            TopicArn=sns_topic_arn,
            Subject=f"FCMates EC2 복구 완료 - {instance_id}",
            Message=json.dumps(message, indent=2, ensure_ascii=False)
        )
        
        print("EC2 복구 프로세스 완료")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'EC2 recovery completed successfully',
                'instance_id': instance_id,
                'actions_taken': recovery_actions
            })
        }
        
    except Exception as e:
        error_message = f"EC2 복구 중 오류 발생: {str(e)}"
        print(error_message)
        
        # 오류 알림 전송
        if sns_topic_arn:
            sns.publish(
                TopicArn=sns_topic_arn,
                Subject=f"FCMates EC2 복구 실패 - {instance_id}",
                Message=json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "instance_id": instance_id,
                    "error": error_message,
                    "event": event
                }, indent=2, ensure_ascii=False)
            )
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': error_message,
                'instance_id': instance_id
            })
        }