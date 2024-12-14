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

- Evidence of the created lambda function:

![image](screenshots\9-lambdaFunction.png)

## Security and testing

- The permission provided to the lambda function was the full all access. 

![image](screenshots\6-lambdaPermissions.png)

- Permission justification:

I didn't want to invest to much time seting up the scope of the access for the lambda function, but by giving full access to Amazon SageMaker from a Lambda function is considered a bad practice due to several key reasons. It poses security risks as it can expose resources to unauthorized usage and increase the attack surface if the function is compromised. This can also lead to unforeseen cost implications from unintended operations. Additionally, adhering to the principle of least privilege is crucial; restricting permissions to only what's necessary minimizes potential damage. Comprehensive auditing and compliance become challenging with broad access, while limiting permissions helps maintain operational control over resources. Overall, implementing specific permissions for the Lambda function enhances security and efficiency.

- Test result:

![image](screenshots\7-test.png)

## Concurrency and auto-scaling

- Concurrency configuration for dummy version:

![image](screenshots\8-Concurrency.png)

- About Lambda function concurrency configuration:
 
Includes reserved concurrency to guarantee a specific number of concurrent executions for a function, ensuring capacity even under high demand. It is subject to account-level limits, which can be increased upon request. Provisioned concurrency allows for pre-allocated instances to reduce cold start times, and throttling occurs when incoming requests exceed the defined limits, resulting in "429 Too Many Requests" errors which require proper handling in applications.

- About Auto-Scaling for Deployed SageMaker Endpoints:

SageMaker allows for auto-scaling of deployed endpoints through automatic scaling policies based on metrics like CPU utilization or memory usage. Users can set minimum and maximum instance capacities to manage resource limits effectively. Scale-out policies dictate how to increase instances during demand spikes, while scale-in policies manage reductions during low usage. A cooldown period can also be configured to stabilize the system before making further scaling decisions, optimizing resource utilization and performance while controlling costs.