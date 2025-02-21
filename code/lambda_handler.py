import boto3
import json

ENDPOINT_NAME = "predict"
runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    try:
        if not isinstance(event, dict):
            raise ValueError("Invalid input format. Must be a JSON object.")

        input_payload = json.dumps(event)

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=input_payload,
        )

        response_body = response["Body"].read().decode("utf-8")

        return {
            "statusCode": 200,
            "body": json.loads(response_body),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }