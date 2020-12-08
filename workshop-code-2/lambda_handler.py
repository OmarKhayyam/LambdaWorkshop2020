#!/usr/bin/env python3

from mylayer import get_input_data,propagate

def handler(event,context):
    b,w,X,Y = get_input_data(event["bucket"],event["key"])
    grads, cost = propagate(w, b, X, Y)
    print ("Derivative with respect to weights dw = " + str(grads["dw"]))
    print ("Derivative with respect to bias db = " + str(grads["db"]))
    print ("Cost = " + str(cost))
