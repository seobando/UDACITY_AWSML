# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling

These are the logs for the training process:

![TrainingJobs](./ScreenShots/TranningLogs/TrainingJobCompleted.png)

![TrainingMetrics](./ScreenShots/TranningLogs/dog-pytorch-2024-08-05-23-57-41-000.png)

There are the results of the 2 tunning jobs:

![TunningJobs](./ScreenShots/TunnerLogs/HyperparametersTuningJobs.png)

![TunningMetric1](./ScreenShots/TunnerLogs/pytorch-training-240805-2136-001-410838b8.png)

![TunningMetric2](./ScreenShots/TunnerLogs/pytorch-training-240805-2136-002-1ea47b7f.png)

### Results

![CrossEntropyLoss](./ScreenShots/CrossEntropyLoss.png)

## Model Deployment

![EndPoint](./ScreenShots/EndPoint.png)
