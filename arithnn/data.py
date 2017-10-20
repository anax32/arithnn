"""Generates random data for binary functions.

Usage:
D = Data ((0, 100)) 
create an object to generate operands in the range 0, 100


Xs, ys = D.create_data (1000, lambda x, y: x + y)
Generate 1000 Xs (pairs of random numbers to use as operands
for the function) and 1000 ys which are the result of
applying the lambda function to these operands.

TODO: return a generator function which can be passed to
keras for the model.train_generator function
"""

from random import randint

class Data (): 
  def __init__ (self, range):
    self.range = range

  def rnd_int (self):
    """returns a random integer in the context range"""
    return randint (self.range[0], self.range[1])
  
  @staticmethod
  def make_safe_fn (fn):
    """returns a function which wraps fn and swallows
    any exceptions generated.
    If an evaluation of fn throws an exception, the
    wrapper will return zero"""
    def a_ (x, y):
      try:
        return fn (x, y)
      except:
        return 0
    return a_
  
  def create_data (self, count, fn):
    """create count number of random operations and the
    correct answers by applying fn to the random operands.
    NB: if fn throws exceptions for particular inputs
    (i.e., divide function and a zero denominator)
    the exception will propagate up; to avoid this fn is
    wrapped using the make_safe_fn function above."""
    fn_ = Data.make_safe_fn (fn)
    X = [(self.rnd_int (), self.rnd_int ()) for x in range (count)]
    y = [fn_ (x[0], x[1]) for x in X]
    return (X, y)  