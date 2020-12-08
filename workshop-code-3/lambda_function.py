#!/usr/bin/env python3

import os
import boto3
import json
import numpy as np
from zipfile import ZipFile
from datetime import datetime

def process_input_files(event,context):
    '''
    This function expects input of type as follows:
    event["bucket"] = <some value>
    event["key"] = <object key>
    '''
    bkt = event["bucket"]
    key = event["key"]
    filename = "inputdata.zip"
    s3 = boto3.client("s3")
    s3.download_file(bkt,key,"/tmp/" + filename)
    with ZipFile("/tmp/" + filename,'r') as zip:
        print("Extracting all files from zip bundle...")
        os.chdir("/tmp")
        zip.extractall() # We now have with us b.npz,w.npz,X.npz and Y.npz
        os.chdir("/")
        print("Done extracting files from zip bundle.")
    s3.upload_file("/tmp/b.npz",bkt,"lambda_input/b.npz")
    s3.upload_file("/tmp/w.npz",bkt,"lambda_input/w.npz")
    s3.upload_file("/tmp/X.npz",bkt,"lambda_input/X.npz")
    s3.upload_file("/tmp/Y.npz",bkt,"lambda_input/Y.npz")
    ## Load the files we have now
    b = { "bucket": bkt, "key": "lambda_input/b.npz" }
    w = { "bucket": bkt, "key": "lambda_input/w.npz" }
    X = { "bucket": bkt, "key": "lambda_input/X.npz" }
    Y = { "bucket": bkt, "key": "lambda_input/Y.npz" }
    result = { "b": {"bucket": bkt, "key": b["key"]}, "w": {"bucket": bkt, "key": w["key"]}, "X": {"bucket": bkt, "key": X["key"]}, "Y": {"bucket": bkt, "key": Y["key"]} }
    # Debug
    print(b,w,X,Y)
    return json.dumps(result)

def initiator(event,context):
    client = boto3.client("stepfunctions")
    mydict = { 'bucket': event['queryStringParameters']['bucket'],'key': event['queryStringParameters']['key']}
    now = datetime.now()
    timestr = now.strftime("-%Y-%m-%d-%H-%M-%S")
    response = client.start_execution(stateMachineArn="arn:aws:states:ap-south-1:684473352813:stateMachine:costcompute1",
                            name="rns-test"+timestr,input=json.dumps(mydict),traceHeader="rns-")
    mydict['executionArn'] = response['executionArn']
    responseBody = { 'message': mydict, 'input': event}
    respons = {
                'statusCode': 200,
                'body': json.dumps(responseBody, separators=(',',':'))
    }
    return respons

def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s

def apply_non_linearity(event,context): ## w, b, X, Y
    s3 = boto3.client("s3")
    print("Event: {}".format(str(json.loads(event))))
    event = json.loads(event)
    s3.download_file(event["w"]["bucket"],event["w"]["key"],"/tmp/" + "w.npz")
    s3.download_file(event["X"]["bucket"],event["X"]["key"],"/tmp/" + "X.npz")
    s3.download_file(event["b"]["bucket"],event["b"]["key"],"/tmp/" + "b.npz")
    s3.download_file(event["Y"]["bucket"],event["Y"]["key"],"/tmp/" + "Y.npz")
    w = np.load("/tmp/w.npz")["arr_0"]
    X = np.load("/tmp/X.npz")["arr_0"]
    b = np.load("/tmp/b.npz")["arr_0"]
    Y = { "bucket": event["Y"]["bucket"], "key": event["Y"]["key"] }
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)
    np.savez("/tmp/A.npz",A)
    s3.upload_file("/tmp/A.npz",event["w"]["bucket"],"lambda_input/A.npz")
    result = {"A": {"bucket": event["w"]["bucket"], "key": "lambda_input/A.npz"}, "Y": {"bucket": Y["bucket"], "key": Y["key"]}, "m": m }
    # Debug
    print(A,Y,m)
    return json.dumps(result)

def compute_cost_1(event,context):
    s3 = boto3.client("s3")
    event = json.loads(event)
    s3.download_file(event["A"]["bucket"],event["A"]["key"],"/tmp/" + "A.npz")
    s3.download_file(event["Y"]["bucket"],event["Y"]["key"],"/tmp/" + "Y.npz")
    A = np.load("/tmp/A.npz")["arr_0"]
    Y = np.load("/tmp/Y.npz")["arr_0"]
    m = event["m"]
    cost_1 = np.multiply(Y,np.log(A))
    np.savez("/tmp/cost_1.npz",cost_1)
    s3.upload_file("/tmp/cost_1.npz",event["A"]["bucket"],"lambda_input/cost_1.npz")
    result = {"cost": {"bucket": event["A"]["bucket"],"key": "lambda_input/cost_1.npz"}, "m": m}
    return json.dumps(result)

def compute_cost_2(event,context):
    s3 = boto3.client("s3")
    event = json.loads(event)
    s3.download_file(event["A"]["bucket"],event["A"]["key"],"/tmp/" + "A.npz")
    s3.download_file(event["Y"]["bucket"],event["Y"]["key"],"/tmp/" + "Y.npz")
    A = np.load("/tmp/A.npz")["arr_0"]
    Y = np.load("/tmp/Y.npz")["arr_0"]
    m = event["m"]
    cost_2 = np.multiply((1 - Y),np.log(1 - A))
    np.savez("/tmp/cost_2.npz",cost_2)
    s3.upload_file("/tmp/cost_2.npz",event["A"]["bucket"],"lambda_input/cost_2.npz")
    result = {"cost": {"bucket": event["A"]["bucket"],"key": "lambda_input/cost_2.npz"}, "m": m}
    return json.dumps(result)

def consolidate_cost(event,context):
    s3 = boto3.client("s3")
    count = 0
    m = 0
    for i in event:
        element = json.loads(i)
        s3.download_file(element["cost"]["bucket"],element["cost"]["key"],"/tmp/" + "cost_"+ str(count)+".npz")
        m = element["m"]
        count = count + 1
    #s3.download_file(event[0]["cost_1"]["bucket"],event[0]["cost_1"]["key"],"/tmp/" + "cost_1.npz")
    #s3.download_file(event[1]["cost_2"]["bucket"],event[1]["cost_2"]["key"],"/tmp/" + "cost_2.npz")
    cost_1 = np.load("/tmp/cost_0.npz")["arr_0"]
    cost_2 = np.load("/tmp/cost_1.npz")["arr_0"]
    cost = (-1)*np.sum(cost_1 + cost_2)/m
    cost = np.squeeze(cost)
    return cost
