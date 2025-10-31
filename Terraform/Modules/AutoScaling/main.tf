# Launch Template을 위한 최신 AMI 조회
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# Security Group for Auto Scaling instances
resource "aws_security_group" "autoscaling_sg" {
  name_prefix = "${var.name_prefix}-asg-"
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr_blocks]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-asg-security-group"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# Auto Scaling Group
module "autoscaling" {
  source = "terraform-aws-modules/autoscaling/aws"

  # Auto Scaling Group
  name             = var.name_prefix
  use_name_prefix  = true
  instance_name    = "${var.name_prefix}-instance"

  min_size                  = var.min_size
  max_size                  = var.max_size
  desired_capacity          = var.desired_capacity
  wait_for_capacity_timeout = 0
  default_instance_warmup   = 300
  health_check_type         = "EC2"
  health_check_grace_period = 300

  # Subnet configuration
  vpc_zone_identifier = var.subnet_ids

  # Scaling policies
  scaling_policies = {
    avg-cpu-policy-up = {
      policy_type               = "TargetTrackingScaling"
      estimated_instance_warmup = 1200
      target_tracking_configuration = {
        predefined_metric_specification = {
          predefined_metric_type = "ASGAverageCPUUtilization"
        }
        target_value = var.target_cpu_utilization
      }
    }
  }

  # Instance configuration
  launch_template_name        = "${var.name_prefix}-lt"
  launch_template_description = "Launch template for ${var.name_prefix}"
  update_default_version      = true

  image_id          = var.ami_id != null ? var.ami_id : data.aws_ami.amazon_linux.id
  instance_type     = var.instance_type
  key_name          = var.key_name
  ebs_optimized     = var.ebs_optimized
  enable_monitoring = var.enable_monitoring

  security_groups = [aws_security_group.autoscaling_sg.id]

  # IAM instance profile
  create_iam_instance_profile = true
  iam_role_name               = "${var.name_prefix}-asg-iam-role"
  iam_role_path               = "/ec2/"
  iam_role_description        = "IAM role for ${var.name_prefix} Auto Scaling instances"
  iam_role_tags = merge(var.tags, {
    Name = "${var.name_prefix}-asg-iam-role"
  })
  
  iam_role_policies = {
    AmazonSSMManagedInstanceCore = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    CloudWatchAgentServerPolicy  = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
    S3AccessPolicy              = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
  }

  # Block device mappings
  block_device_mappings = [
    {
      device_name = "/dev/xvda"
      no_device   = 0
      ebs = {
        delete_on_termination = true
        encrypted             = true
        volume_size           = var.root_volume_size
        volume_type           = "gp3"
      }
    }
  ]

  # User data
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    region = var.aws_region
  }))

  # Metadata options
  metadata_options = {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 32
    instance_metadata_tags      = "enabled"
  }

  # Tag specifications
  tag_specifications = [
    {
      resource_type = "instance"
      tags = merge(var.tags, {
        Name          = "${var.name_prefix}-asg-instance"
        BackupEnabled = "true"
        Environment   = var.environment
      })
    },
    {
      resource_type = "volume"
      tags = merge(var.tags, {
        Name        = "${var.name_prefix}-asg-volume"
        Environment = var.environment
      })
    },
    {
      resource_type = "network-interface"
      tags = merge(var.tags, {
        Name        = "${var.name_prefix}-asg-eni"
        Environment = var.environment
      })
    }
  ]

  tags = merge(var.tags, {
    Name        = "${var.name_prefix}-autoscaling-group"
    Environment = var.environment
  })
}

# Auto Scaling Notifications
resource "aws_autoscaling_notification" "asg_notifications" {
  count = var.enable_notifications ? 1 : 0

  group_names = [module.autoscaling.autoscaling_group_name]

  notifications = [
    "autoscaling:EC2_INSTANCE_LAUNCH",
    "autoscaling:EC2_INSTANCE_TERMINATE",
    "autoscaling:EC2_INSTANCE_LAUNCH_ERROR",
    "autoscaling:EC2_INSTANCE_TERMINATE_ERROR",
  ]

  topic_arn = var.sns_topic_arn
}