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
    
    dataset = ""
    dataset_filename = ""
    modelFolder = ""
    #check for dataset 
    if "dataset" in event:
      dataset = event["dataset"]
    else:
        raise Exception("requires csv dataset in event")
    #check for dataset file name
    if "dataset_filename" in event:
      dataset_filename = event["dataset_filename"]
    else:
        raise Exception("requires csv dataset filename in event")
    #check for model name (name of folder to store model artifacts, progress.txt, datasets) 
    if "modelFolder" in event:
      modelFolder = event["modelFolder"]
    else:
        raise Exception("requires model folder name in event")
    
    print(f"dataset: {dataset}, dataset_filename: {dataset_filename} modelFolder: {modelFolder}")

    upload_file_path = os.path.join(modelFolder,dataset_filename)
       
    base64_bytes = dataset.encode()       
    bytes = base64.b64decode(base64_bytes)

    #upload csv file to bucket
    key = ""
    filelist=""
    local_file_path = "/tmp/" + dataset_filename
    with open(local_file_path, 'wb') as fp:
      fp.write(bytes)
      bucket.upload_file(local_file_path, upload_file_path)

    return {
      'statusCode': 200,
      'body': "Uploaded"
    }
    
  # on an error, try to upload error message to S3:
  except Exception as err:
    print("**ERROR**")
    print(str(err))

    return {
      'statusCode': 400,
      'body': json.dumps(str(err))
    }
