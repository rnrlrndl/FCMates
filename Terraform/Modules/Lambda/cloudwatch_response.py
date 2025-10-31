import json
import boto3
import os
from datetime import datetime, timedelta

def lambda_handler(event, context):
    """
    CloudWatch 메트릭 이상 상황 대응 Lambda 함수
    메트릭 임계값 초과 시 자동으로 대응 조치를 취합니다.
    """
    
    # AWS 클라이언트 초기화
    cloudwatch = boto3.client('cloudwatch', region_name=os.environ.get('AWS_REGION', 'ap-northeast-2'))
    ec2 = boto3.client('ec2', region_name=os.environ.get('AWS_REGION', 'ap-northeast-2'))
    autoscaling = boto3.client('autoscaling', region_name=os.environ.get('AWS_REGION', 'ap-northeast-2'))
    sns = boto3.client('sns', region_name=os.environ.get('AWS_REGION', 'ap-northeast-2'))
    
    # 환경 변수에서 설정 값 가져오기
    instance_id = os.environ.get('EC2_INSTANCE_ID')
    sns_topic_arn = os.environ.get('SNS_TOPIC_ARN')
    
    try:
        print("CloudWatch 메트릭 대응 프로세스 시작")
        print(f"이벤트 상세: {json.dumps(event, indent=2)}")
        
        # 이벤트에서 알람 정보 추출
        alarm_name = event.get('detail', {}).get('alarmName', '')
        metric_name = event.get('detail', {}).get('metricName', '')
        state_value = event.get('detail', {}).get('state', {}).get('value', '')
        
        print(f"알람: {alarm_name}, 메트릭: {metric_name}, 상태: {state_value}")
        
        response_actions = []
        
        # 알람 상태가 ALARM이 아니면 처리하지 않음
        if state_value != 'ALARM':
            print(f"알람 상태가 ALARM이 아닙니다: {state_value}")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'No action needed - alarm not in ALARM state'})
            }
        
        # 1. CPU 사용률 높음 대응
        if 'cpu' in alarm_name.lower() or metric_name == 'CPUUtilization':
            print("CPU 사용률 높음 알람 감지 - 대응 조치 시작")
            
            # 현재 CPU 사용률 확인
            cpu_stats = cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=datetime.utcnow() - timedelta(minutes=15),
                EndTime=datetime.utcnow(),
                Period=300,
                Statistics=['Average', 'Maximum']
            )
            
            if cpu_stats['Datapoints']:
                latest_cpu = cpu_stats['Datapoints'][-1]['Average']
                max_cpu = max([dp['Maximum'] for dp in cpu_stats['Datapoints']])
                
                print(f"현재 CPU 평균: {latest_cpu}%, 최대: {max_cpu}%")
                
                # CPU 사용률이 매우 높은 경우 (95% 이상)
                if max_cpu >= 95:
                    print("위험 수준의 CPU 사용률 - 인스턴스 재부팅")
                    ec2.reboot_instances(InstanceIds=[instance_id])
                    response_actions.append(f"긴급 재부팅 (CPU {max_cpu}%)")
                
                # Auto Scaling Group이 있다면 스케일 아웃 시도
                try:
                    # 인스턴스가 속한 ASG 찾기
                    asg_response = autoscaling.describe_auto_scaling_instances(
                        InstanceIds=[instance_id]
                    )
                    
                    if asg_response['AutoScalingInstances']:
                        asg_name = asg_response['AutoScalingInstances'][0]['AutoScalingGroupName']
                        
                        # ASG 정보 가져오기
                        asg_details = autoscaling.describe_auto_scaling_groups(
                            AutoScalingGroupNames=[asg_name]
                        )
                        
                        if asg_details['AutoScalingGroups']:
                            asg = asg_details['AutoScalingGroups'][0]
                            current_capacity = asg['DesiredCapacity']
                            max_capacity = asg['MaxSize']
                            
                            # 용량 증가 가능한 경우
                            if current_capacity < max_capacity:
                                new_capacity = min(current_capacity + 1, max_capacity)
                                autoscaling.set_desired_capacity(
                                    AutoScalingGroupName=asg_name,
                                    DesiredCapacity=new_capacity,
                                    HonorCooldown=False
                                )
                                response_actions.append(f"ASG 스케일 아웃: {current_capacity} → {new_capacity}")
                                print(f"Auto Scaling Group {asg_name} 용량 증가: {new_capacity}")
                
                except Exception as e:
                    print(f"Auto Scaling 작업 중 오류: {str(e)}")
        
        # 2. 메모리 사용률 높음 대응
        elif 'memory' in alarm_name.lower() or metric_name == 'mem_used_percent':
            print("메모리 사용률 높음 알람 감지")
            
            # 메모리 최적화를 위한 CloudWatch 커스텀 메트릭 생성
            cloudwatch.put_metric_data(
                Namespace='FCMates/Recovery',
                MetricData=[
                    {
                        'MetricName': 'MemoryCleanupTriggered',
                        'Value': 1,
                        'Unit': 'Count',
                        'Dimensions': [
                            {
                                'Name': 'InstanceId',
                                'Value': instance_id
                            }
                        ]
                    }
                ]
            )
            
            response_actions.append("메모리 정리 신호 전송")
        
        # 3. 디스크 사용률 높음 대응
        elif 'disk' in alarm_name.lower():
            print("디스크 사용률 높음 알람 감지")
            
            # S3 정리 Lambda 함수 호출을 위한 이벤트 생성
            lambda_client = boto3.client('lambda', region_name=os.environ.get('AWS_REGION', 'ap-northeast-2'))
            
            try:
                lambda_client.invoke(
                    FunctionName=f"{os.environ.get('NAME_PREFIX', 'fcmates')}-s3-cleanup",
                    InvocationType='Event',  # 비동기 호출
                    Payload=json.dumps({
                        'trigger': 'disk_space_alarm',
                        'instance_id': instance_id
                    })
                )
                response_actions.append("S3 정리 작업 비동기 호출")
                
            except Exception as e:
                print(f"S3 정리 Lambda 호출 실패: {str(e)}")
        
        # 4. 네트워크 관련 대응
        elif 'network' in alarm_name.lower():
            print("네트워크 관련 알람 감지")
            
            # 네트워크 메트릭 수집
            network_metrics = ['NetworkIn', 'NetworkOut', 'NetworkPacketsIn', 'NetworkPacketsOut']
            
            for metric in network_metrics:
                try:
                    stats = cloudwatch.get_metric_statistics(
                        Namespace='AWS/EC2',
                        MetricName=metric,
                        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                        StartTime=datetime.utcnow() - timedelta(minutes=10),
                        EndTime=datetime.utcnow(),
                        Period=300,
                        Statistics=['Average', 'Maximum']
                    )
                    
                    if stats['Datapoints']:
                        latest_value = stats['Datapoints'][-1]['Average']
                        print(f"{metric}: {latest_value}")
                        
                except Exception as e:
                    print(f"네트워크 메트릭 {metric} 수집 실패: {str(e)}")
            
            response_actions.append("네트워크 상태 점검 완료")
        
        # 5. 종합 상황 보고서 생성
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "alarm_details": {
                "name": alarm_name,
                "metric": metric_name,
                "state": state_value
            },
            "instance_id": instance_id,
            "response_actions": response_actions,
            "event_source": event.get('source', ''),
            "region": os.environ.get('AWS_REGION', 'ap-northeast-2')
        }
        
        # 6. SNS 알림 전송
        sns.publish(
            TopicArn=sns_topic_arn,
            Subject=f"FCMates CloudWatch 알람 대응 완료 - {alarm_name}",
            Message=json.dumps(report, indent=2, ensure_ascii=False)
        )
        
        print("CloudWatch 메트릭 대응 완료")
        print(f"수행된 작업: {response_actions}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'CloudWatch alarm response completed',
                'alarm_name': alarm_name,
                'actions_taken': response_actions
            })
        }
        
    except Exception as e:
        error_message = f"CloudWatch 메트릭 대응 중 오류 발생: {str(e)}"
        print(error_message)
        
        # 오류 알림 전송
        if sns_topic_arn:
            sns.publish(
                TopicArn=sns_topic_arn,
                Subject=f"FCMates CloudWatch 알람 대응 실패",
                Message=json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": error_message,
                    "event": event
                }, indent=2, ensure_ascii=False)
            )
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': error_message
            })
        }