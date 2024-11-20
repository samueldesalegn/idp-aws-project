import boto3
import json
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
textract_client = boto3.client('textract')
comprehend_client = boto3.client('comprehend')


def lambda_handler(event, context):
    """
    Main Lambda function to process documents uploaded to S3.
    """
    try:
        # Iterate over S3 event records
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            logger.info(f"Processing file: s3://{bucket}/{key}")

            # Extract text and tables from the document using Textract
            textract_response = process_with_textract(bucket, key)

            # Parse the Textract response to extract key-value pairs
            extracted_data = parse_textract_response(textract_response)
            logger.info(f"Extracted Data: {json.dumps(extracted_data)}")

            # Optional: Analyze extracted text with Comprehend
            comprehend_response = analyze_with_comprehend(extracted_data)
            logger.info(f"Comprehend Entities: {json.dumps(comprehend_response)}")

            # Save the processed results back to S3
            save_results_to_s3(bucket, key, extracted_data, comprehend_response)

        return {
            "statusCode": 200,
            "body": json.dumps("Document processed successfully!")
        }
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error: {str(e)}")
        }


def process_with_textract(bucket, key):
    """
    Use Amazon Textract to analyze the document.
    """
    response = textract_client.analyze_document(
        Document={'S3Object': {'Bucket': bucket, 'Name': key}},
        FeatureTypes=['TABLES', 'FORMS']
    )
    return response


def parse_textract_response(response):
    """
    Extract key-value pairs from Textract response.
    """
    extracted_data = {}
    for block in response['Blocks']:
        if block['BlockType'] == 'KEY_VALUE_SET' and 'EntityTypes' in block:
            if 'KEY' in block['EntityTypes']:
                key = block.get('Text', '').strip()
                value_block = next(
                    (b for b in response['Blocks'] if b['Id'] in block.get('Relationships', [{}])[0].get('Ids', [])),
                    {}
                )
                value = value_block.get('Text', '').strip()
                extracted_data[key] = value
    return extracted_data


def analyze_with_comprehend(extracted_data):
    """
    Use Amazon Comprehend to analyze extracted text for entities.
    """
    text = "\n".join(f"{k}: {v}" for k, v in extracted_data.items())
    response = comprehend_client.detect_entities(
        Text=text,
        LanguageCode='en'
    )
    return response['Entities']


def save_results_to_s3(bucket, key, textract_data, comprehend_data):
    """
    Save the processed results back to S3 as a JSON file.
    """
    result_key = key.replace('.pdf', '_results.json')
    results = {
        'TextractData': textract_data,
        'ComprehendEntities': comprehend_data
    }
    s3_client.put_object(
        Bucket=bucket,
        Key=result_key,
        Body=json.dumps(results)
    )
    logger.info(f"Results saved to: s3://{bucket}/{result_key}")
