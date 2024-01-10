import requests
import pathlib
import logging
import sys
import base64

from configparser import ConfigParser

#
# prompt
#
def prompt():
  """
  Prompts the user and returns the command number

  Parameters
  ----------
  None

  Returns
  -------
  Command number entered by user (0, 1, 2, ...)
  """
  print()
  print(">> Enter a command:")
  print("   0 => End")
  print("   1 => Start a training session")
  print("   2 => Inference from a model")

  cmd = input()

  if cmd == "":
    cmd = -1
  elif not cmd.isnumeric():
    cmd = -1
  else:
    cmd = int(cmd)

  return cmd

def training_session(baseurl):
  """
  Starts a training session.

  Parameters
  ----------
  baseurl: baseurl for web service

  Returns
  -------
  nothing
  """

  try:
    print()
    print(">> Enter a command:")
    print("   0 => Continue a training session")
    print("   1 => Instantiate a new training session")
    new_train = int(input())
    sess_name = "" #this will be the name of the folder containing model artifacts
    if(new_train):
      print()
      print("Type in a unique name of your new session: ")
      sess_name = input()
    else:
      print()
      print("Type in the name of the session you wish to continue: ")
      sess_name = input()
    
    print()
    print(">> Enter a command:")
    print("   0 => Classification task")
    print("   1 => Regression task")
    train_type = int(input())

    print()
    print("Before training, you need to upload a dataset.")
    print("Type in the name of your csv dataset: ")
    dataset_filename = input()

    while dataset_filename[-3:] != "csv":
      print("Enter a csv file.")
      dataset_filename = input()

    while(not pathlib.Path(dataset_filename).is_file()):
       print("csv file '", dataset_filename, "' does not exist...")
       print()
       print("Type in the name of your csv dataset.")
       dataset_filename = input()

    print()
    print(f"Uploading {dataset_filename} to {sess_name}/ ...")
    upload(baseurl, dataset_filename, sess_name)

    if not train_type: #classification
        print()
        print("Training your classification model...")
        data = {"datasetFilenameIn": dataset_filename, "firstModel": new_train, "modelBucket": sess_name}
        url = baseurl + '/trainSGDClassifier'
        res = requests.get(url, json=data)

        if res.status_code != 200:
            print("Failed with status code:", res.status_code)
            print("url: " + url)
            if res.status_code == 400:
                body = res.json()
                print("Error message:", body)
            return

        body = res.json()
        print(f"SGDClassifier trained and achieved accuracy of {body}")
    
    else: #regression
       print()
       print("Training your regression model...")
       data = {"datasetFilenameIn": dataset_filename, "firstModel": new_train, "modelBucket": sess_name}
       url = baseurl + '/trainSGDRegressor'
       res = requests.get(url, json=data)
       
       if res.status_code != 200:
          print("Failed with status code:", res.status_code)
          print("url: " + url)
          if res.status_code == 400:
              body = res.json()
              print("Error message:", body)
          return
       body = res.json()
       print(f"SGDRegressor trained and achieved R2 score of {body}")

  except Exception as e:
    logging.error("training_session() failed:")
    logging.error("url: " + url)
    logging.error(e)
    return


############################################################
def upload(baseurl, dataset_filename, sess_name):
  """
  Uploads a dataset to s3.

  Parameters
  ----------
  baseurl: baseurl for web service
  dataset_filename: name of dataset with .csv suffix
  sess_name: name used for model bucket

  Returns
  -------
  nothing
  """
  try:
      infile = open(dataset_filename, "rb")
      bytes = infile.read()
      infile.close()

      data = base64.b64encode(bytes)
      datastr = data.decode()

      data = {"dataset_filename": dataset_filename, "dataset": datastr, "modelFolder": sess_name}

      url = baseurl + '/uploadDataset'
      res = requests.post(url, json=data)

      if res.status_code != 200:
          print("Failed with status code:", res.status_code)
          print("url: " + url)
          if res.status_code == 400:
              body = res.json()
              print("Error message:", body)
          return

      print("Dataset uploaded")
      return

  except Exception as e:
      logging.error("upload() failed:")
      logging.error("url: " + url)
      logging.error(e)
      return


############################################################
def inference(baseurl):
  """
  Starts an inference session to get predictions from model
  that user chooses.

  Parameters
  ----------
  baseurl: baseurl for web service

  Returns
  -------
  nothing
  """
  try:
    print()
    print("Type the training session from which you want to infer: ")
    sess_name = input()

    print()
    print(">> Enter your model type:")
    print("   0 => Classifier")
    print("   1 => Regressor")
    model_type = input()

    print()
    print("Getting list of models and their performance metrics...")
    print()
    data = {"modelFolder": sess_name}

    url = baseurl + '/getProgress'
    res = requests.get(url, json=data)

    if res.status_code != 200:
          print("Failed with status code:", res.status_code)
          print("url: " + url)
          if res.status_code == 400:
              body = res.json()
              print("Error message:", body)
          return
    
    body = res.json()

    datastr = body

    base64_bytes = datastr.encode()
    bytes = base64.b64decode(base64_bytes)
    results = bytes.decode()
    print(results)

    print()
    print(">> Type the name of the model you would like to infer from:")
    modelName = input()

    print()
    print("Before inferring, you need to upload a dataset.")
    print("Type in the name of your csv dataset: ")
    dataset_filename = input()

    while dataset_filename[-3:] != "csv":
      print("Enter a csv file.")
      dataset_filename = input()

    while(not pathlib.Path(dataset_filename).is_file()):
       print("csv file '", dataset_filename, "' does not exist...")
       print()
       print("Type in the name of your csv dataset.")
       dataset_filename = input()

    print()
    print(f"Uploading {dataset_filename} to {sess_name}/ ...")
    upload(baseurl, dataset_filename, sess_name)

    print()
    print(f"Inferring from model {modelName}...")
    data = {"datasetFilenameIn": dataset_filename, "modelName": modelName, "modelFolder": sess_name}
    url = baseurl + '/inferModel'
    res = requests.get(url, json=data)
    
    if res.status_code != 200:
      print("Failed with status code:", res.status_code)
      print("url: " + url)
      if res.status_code == 400:
          body = res.json()
          print("Error message:", body)
      return
    
    body = res.json()
    if(not int(model_type)): #classifier
      print(f"{modelName} achieved an accuracy of {body}")
    else: #regressor
      print(f"{modelName} achieved a R2 score of {body}")


  except Exception as e:
    logging.error("inference() failed:")
    logging.error("url: " + url)
    logging.error(e)
    return

############################################################
# main
try:
  print('** Welcome to Incremental Learning Experimenter **')
  print()

  # eliminate traceback so we just get error message:
  sys.tracebacklimit = 0

  config_file = 'inc_learn_config.ini'
  if not pathlib.Path(config_file).is_file():
    print("**ERROR: config file '", config_file, "' does not exist, exiting")
    sys.exit(0)

  configur = ConfigParser()
  configur.read(config_file)
  baseurl = configur.get('client', 'webservice')
  if len(baseurl) < 16:
    print("**ERROR: baseurl '", baseurl, "' is not nearly long enough...")
    sys.exit(0)

  lastchar = baseurl[len(baseurl) - 1]
  if lastchar == "/":
    baseurl = baseurl[:-1]

  
  # main processing loop:
  cmd = prompt()

  while cmd != 0:
    if cmd == 1:
      training_session(baseurl)
    elif cmd == 2:
      inference(baseurl)
    else:
      print("** Unknown command, try again...")
    cmd = prompt()

  print()
  print('** done **')
  sys.exit(0)

except Exception as e:
  logging.error("**ERROR: main() failed:")
  logging.error(e)
  sys.exit(0)