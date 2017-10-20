import numpy as np
from random import random as rnd
import sys

from inspect import signature

from arithnn.arithmodel import Arithmodel
  
def train_adder ():
  """train a network to learn addition"""
  mdl = Arithmodel ("add", '+')
  mdl.train (lambda x, y : x+y)
  
def train_multiplier ():
  """train a network to learn multiplication"""
  mdl = Arithmodel ("mul", 'x')
  mdl.train (lambda x, y : x*y)

def train_subtractor ():
  """train a network to learn subtraction"""
  mdl = Arithmodel ("sub", '-')
  mdl.train (lambda x, y : x-y)
  
def train_modulo ():
  """train a network to learn modulus"""
  mdl = Arithmodel ("mod", '+')
  mdl.train (lambda x, y : x%y)
  
def train_division ():
  """train a network to learn division"""
  mdl = Arithmodel ("div", '/')
  mdl.train (lambda x, y : x/y)
  
def train_all ():
  """train networks to learn addition, multiplication,
  subtraction, modulus and division"""
  train_adder ()
  train_multiplier ()
  train_subtractor ()
  train_modulo ()
  train_division ()
  
def predict_adder (x, y):
  """load a model from "add.hdf5" and predict the response
  for inputs (x, y) i.e., add them together and get the answer"""
  mdl = Arithmodel ("add", '+')
  print (mdl.predict (x, y))
  
def predict_multiplier (x, y):
  """load a model from "mul.hdf5" and predict the response
  for inputs (x, y) i.e., multiply them together and get the answer"""
  mdl = Arithmodel ("mul", '*')
  print (mdl.predict (x, y))

if __name__=="__main__":
  """entry point"""
  ops = {
    "train_add" : train_adder,
    "train_mul" : train_multiplier,
    "train_sub" : train_subtractor,
    "train_mod" : train_modulo,
    "train_div" : train_division,
    "train_all" : train_all,
    "run_add" : predict_adder,
    "run_mul" : predict_multiplier,
#    "run_sub" : predict_subtractor,
#    "run_mod" : predict_modulo,
#    "run_div" : predict_division
  }
  
  try:
    ops[sys.argv[1]] (*sys.argv[2:])
  except KeyError:
    print (__file__ + " : ")
    for o in ops.keys ():
      sig = signature (ops[o])
      print ("  %s %s (%d parameters)," % (o, " ".join (sig.parameters), len (sig.parameters)))