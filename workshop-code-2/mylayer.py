#!/usr/bin/env python3

import os
import boto3
import numpy as np
from zipfile import ZipFile

def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)
    cost = (-1)*np.sum(np.multiply(Y,np.log(A)) + np.multiply((1 - Y),np.log(1 - A)))/m
    dw = np.dot(X,(A - Y).T)/m
    db = np.sum((A - Y))/m
    cost = np.squeeze(cost)
    grads = {"dw": dw,"db": db}
    return grads, cost

def get_input_data(bucket,ky):
    bkt = bucket
    key = ky
    filename = "inputdata.zip"
    s3 = boto3.client("s3")
    s3.download_file(bkt,key,"/tmp/" + filename)
    with ZipFile("/tmp/" + filename,'r') as zip:
        print("Extracting all files from zip bundle...")
        os.chdir("/tmp")
        zip.extractall() # We now have with us b.npz,w.npz,X.npz and Y.npz
        os.chdir("/")
        print("Done extracting files from zip bundle.")
    ## Load the files we have now
    b = np.load("/tmp/" + "b.npz")["arr_0"]
    w = np.load("/tmp/" + "w.npz")["arr_0"]
    X = np.load("/tmp/" + "X.npz")["arr_0"]
    Y = np.load("/tmp/" + "Y.npz")["arr_0"]
    return b,w,X,Y
