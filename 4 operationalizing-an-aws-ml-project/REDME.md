# Project: Operationalizing an AWS ML Project



## Training and deployment

- Intance selected was ml.t3.medium:

![image](screenshots\1-notebook-instance.png)

- Instance choice justification:

This instance was selected because it is cost-effective, has burstable performance, provides sufficient resources for development, offers flexibility due to being part of the T3 instance family, and is suitable for general purposes.

- S3 buckets containing data and models

![image](screenshots\2-s3.png)

- Record of ran jobs:

![image](screenshots\3-training-jobs.png)

- Endpoint generated:

![image](screenshots\4-endpoint.png)

## EC2 Training

- Instance choice justification:

The instance selected was a 'Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5.1 (Amazon Linux 2023) 20241208' of type m5.xlarge because it provides a balanced combination of computational power, memory, and cost efficiency, making it suitable for training and deploying machine learning models. The m5.xlarge instance offers 4 vCPUs and 16 GiB of memory, which is ideal for handling larger datasets and more complex model architectures. Additionally, it supports NVIDIA GPUs, which enhance performance for deep learning tasks that leverage GPU acceleration, making it an optimal choice for workloads that require both high performance and scalability.

- Evidence of generated model:

![image](screenshots\5-ec2-model.png)

## Lambda function setup

![image](screenshots\9-lambdaFunction.png)

## Security and testing

![image](screenshots\6-lambdaPermissions.png)

![image](screenshots\7-test.png)

## Concurrency and auto-scaling

![image](screenshots\8-Concurrency.png)