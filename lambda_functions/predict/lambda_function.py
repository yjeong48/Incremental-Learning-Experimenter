import json
import boto3
import os
import csv
import tempfile
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler

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
    s3_client = boto3.client('s3')
    
    modelFolder = ""
    modelName = ""
    datasetFilenameIn = ""
    trainType = ""
    #check for name of folder to store model artifacts, progress.txt, datasets
    if "modelFolder" in event:
      modelFolder = event["modelFolder"]
    else:
        raise Exception("requires model folder name in event")
    #check for model name
    if "modelName" in event:
      modelName = event["modelName"]
    else:
        raise Exception("requires model name in event")
    #check for dataset filename
    if "datasetFilenameIn" in event:
      datasetFilenameIn = event["datasetFilenameIn"]
    else:
        raise Exception("requires dataset file name in event")
    #check for train type (0 for classification, 1 for regression)
    if "trainType" in event:
      trainType = event["trainType"]
    else:
        raise Exception("requires train type in event")

    print(f"modelFolder: {modelFolder}, modelName: {modelName}, datasetFilenameIn: {datasetFilenameIn}, trainType: {trainType}")

    ##get dataset in modelBucket##
    #download to local memory then load
    local_file_path = "/tmp/dataset.csv" 
    dataset_file_path = os.path.join(modelFolder,datasetFilenameIn)
    try:
      bucket.download_file(dataset_file_path, local_file_path)
    except Exception as err:
      print("no dataset found")
      print(str(err))
      return {
        'statusCode': 400,
        'body': json.dumps("No dataset found.")
      }
    X=[]
    y=[]
    

    with open(local_file_path, 'r') as file:
      my_reader = csv.reader(file, delimiter=',')
      next(my_reader)
      for row in my_reader:
          X.append([float(value) for value in row[:-1]])  
          y.append(float(row[-1])) 

    X = np.array(X)
    y = np.array(y)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    if (len(X) == 0):
      print("**error reading csv dataset file, returning...**")
      raise Exception("Could not read dataset.")
    
    #download model to local memory temporarily then load
    model_file_path = os.path.join(modelFolder,modelName)
    print(f"model_file_path is {model_file_path}")

    with tempfile.TemporaryFile() as fp:
      s3_client.download_fileobj(Fileobj=fp, Bucket=bucketname, Key=model_file_path)
      fp.seek(0)
      loaded_model = joblib.load(fp)

    if(trainType):
      accuracy = loaded_model.score(X,y)
      print(f"R2 score: {accuracy}")
    else:
      accuracy = loaded_model.score(X,y)
      print(f"Accuracy: {accuracy}")

    return {
      'statusCode': 200,
      'accuracy': accuracy
    }

    
  # on an error, try to upload error message to S3:
  except Exception as err:
    print("**ERROR**")
    print(str(err))

    return {
      'statusCode': 400,
      'body': json.dumps(str(err))
    }
