import json
import boto3
import os
import base64


from configparser import ConfigParser

def lambda_handler(event, context):
  try:
    print("**STARTING**")
    
    #
    # setup AWS based on config file:
    #
    config_file = 'config.ini'
    os.environ['AWS_SHARED_CREDENTIALS_FILE'] = config_file
    
    configur = ConfigParser()
    configur.read(config_file)
    
    #
    # configure for S3 access:
    #
    s3_profile = 's3readwrite'
    boto3.setup_default_session(profile_name=s3_profile)
    
    bucketname = configur.get('s3', 'bucket_name')
    
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucketname)
    
    modelFolder = ""
    #check for name of folder to store model artifacts, progress.txt, datasets
    if "modelFolder" in event:
      modelFolder = event["modelFolder"]
    else:
        raise Exception("requires model folder name in event")
    
    print(f"modelFolder: {modelFolder}")

    #get progress of models
    local_file_path = "/tmp/" + "progress.txt"
    progress_file_path = os.path.join(modelFolder,"progress.txt")
    bucket.download_file(progress_file_path, local_file_path)

    with open(local_file_path, "rb") as fp:
      bytes = fp.read()
    
    data = base64.b64encode(bytes)
    datastr = data.decode()

    return {
      'statusCode': 200,
      'body': json.dumps(datastr)
    }

    
  # on an error, try to upload error message to S3:
  except Exception as err:
    print("**ERROR**")
    print(str(err))

    return {
      'statusCode': 400,
      'body': json.dumps(str(err))
    }
