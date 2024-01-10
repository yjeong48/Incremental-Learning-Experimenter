import json
import boto3
import os
import uuid
import csv
import tempfile

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import joblib
import numpy as np

from configparser import ConfigParser

def lambda_handler(event):
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
    
    # check for bucket name to store model
    modelBucket = ""
    datasetFilenameIn = ""
    firstModel = 0
    if "modelBucket" in event:
      modelBucket = event["modelBucket"]
    else:
        raise Exception("requires model bucket in event")
    #check for dataset filename
    if "datasetFilenameIn" in event:
      datasetFilenameIn = event["datasetFilenameIn"]
    else:
        raise Exception("requires model fit type in event")
    #check if model has already been trained or not
    if "firstModel" in event:
      firstModel = event["firstModel"]
    else:
        raise Exception("requires model fit type in event")
    
        
    print("modelBucket: ", modelBucket, "firstModel: ", firstModel, "dataset: ", datasetFilenameIn)

    ##get dataset in modelBucket##
    #download to local memory then load
    local_file_path = "/tmp/dataset.csv" 
    dataset_file_path = os.path.join(modelBucket,datasetFilenameIn)
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
    
    if (len(X) == 0):
      print("**error reading csv dataset file, returning...**")
      raise Exception("Could not read dataset.")
    
  # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
      
    modelName = ""
    model_file_path = ""
    prog_file_path = os.path.join(modelBucket,"progress.txt")
    #if firstModel, instatiate new model and save in modelBucket 
    if(firstModel):
      print("first model")
      print(X_train.shape, y_train.shape)
      
      # Initialize the SGDClassifier
      sgd_clf = SGDClassifier(max_iter = 100, loss='log', early_stopping=True, shuffle = True, learning_rate='adaptive', eta0 = 0.01)
      sgd_clf.fit(X_train, y_train)
      accuracy = sgd_clf.score(X_test,y_test)
      print(f"Accuracy: {accuracy}")

      #save model in modelBucket
      modelName = "model" + str(uuid.uuid4()) + ".joblib"
      model_file_path = os.path.join(modelBucket,modelName)
      with tempfile.TemporaryFile() as fp:
        joblib.dump(sgd_clf, fp)
        fp.seek(0)
        s3_client.put_object(Body=fp.read(), Bucket=bucketname, Key=model_file_path)
      
      #create progress.txt file
      s3_client.put_object(Body="", Bucket=bucketname, Key=prog_file_path)

      print("model saved as ", modelName)

    else: #if not firstModel, download latest model in modelBucket and save
      ##get latest model##
      s3_client = boto3.client('s3')
      prefix = os.path.join(modelBucket,"model")
      response = s3_client.list_objects_v2(Bucket=bucketname, Prefix=prefix)
      #print(f"respone is {response}")
      all_models = [obj for obj in response.get('Contents', []) if obj['Key'].endswith("joblib")]
      if(len(all_models) == 0):
         raise Exception("Initial model not found.")
      latest_model = max(all_models, key=lambda x: x['LastModified'])
      print(f"latest model is {latest_model}, this should have .joblib suffix")

      #download to local memory temporarily then load
      model_file_path = latest_model['Key']
      print(f"model_file_path is {model_file_path}")
      with tempfile.TemporaryFile() as fp:
        s3_client.download_fileobj(Fileobj=fp, Bucket=bucketname, Key=model_file_path)
        fp.seek(0)
        loaded_model = joblib.load(fp)

      print("subsequent model")
      loaded_model.learning_rate = 'adaptive'
      loaded_model.eta0 = 0.01
      loaded_model.early_stopping = False

      n_iterations = 100
      batch_size = 32
      #loop to mimic epochs since partial_fit only does one step in gradient descent
      for iteration in range(n_iterations):
          permutation_0 = np.random.permutation(X_train.shape[0])
          X_train = X_train[permutation_0]
          y_train = y_train[permutation_0]

          # Split the training data into batches
          for i in range(0, len(X_train), batch_size):
              X_batch = X_train[i:i+batch_size]
              y_batch = y_train[i:i+batch_size]

              # Perform partial fit on the current batch
              loaded_model.partial_fit(X_batch, y_batch)

      accuracy = loaded_model.score(X_test,y_test)
      print(f"Accuracy: {accuracy}")

      #save model in modelBucket
      modelName = "model" + str(uuid.uuid4()) + ".joblib"
      model_file_path = os.path.join(modelBucket,modelName)
      with tempfile.TemporaryFile() as fp:
        joblib.dump(loaded_model, fp)
        fp.seek(0)
        s3_client.put_object(Body=fp.read(), Bucket=bucketname, Key=model_file_path)

      print("subsequent model saved as ", modelName)

    #write to progress.txt with model name and accuracy
    writeLine = modelName + " " + str(accuracy)
    print(f"writeLine is {writeLine}")
    local_prog_file_path = "/tmp/progress.txt" 
    bucket.download_file(prog_file_path, local_prog_file_path)
    with open(local_prog_file_path, "a+") as fp:
      fp.seek(0)
      data = fp.read(100)
      if len(data) > 0:
         fp.write("\n")
      fp.write(writeLine)
      fp.seek(0)
      s3_client.put_object(Body=fp.read(), Bucket=bucketname, Key=prog_file_path)

    print("wrote model name and acc to progress.txt")
    return {
        'statusCode': 200,
        'accuracy': accuracy
      }

  #
  # on an error, try to upload error message to S3:
  #
  except Exception as err:
    print("**ERROR**")
    print(str(err))
    
    # done, return:
       
    return {
      'statusCode': 400,
      'body': json.dumps(str(err))
    }

def main(): 
  event = {
  "modelBucket": "spam",
  "datasetFilenameIn": "spamdata3.csv",
  "firstModel": 1
}

  lambda_handler(event)
  
  
# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 