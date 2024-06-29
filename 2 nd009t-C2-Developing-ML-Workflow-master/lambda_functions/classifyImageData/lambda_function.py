import json
import boto3
import base64

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2024-06-29-15-21-27-624"

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["body"]['image_data'])

    # Initialize the boto3 client for SageMaker runtime
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    
    # Invoke the endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType='image/png',
        Body=image
    )    
    
    # Make a prediction:
    inferences = response['Body'].read()
    
    # We return the data back to the Step Function    
    inferences_decode = inferences.decode('utf-8')
    
    return {
        'statusCode': 200,
        'body': {
            "inferences":inferences_decode
            
        }
    }