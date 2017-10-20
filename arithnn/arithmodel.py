"""Arithmodel is reponsible for building and training a neural network,
and later using a network for prediction.

To use:
create a model with a name and operator symbol
A = Arithmodel ("multplication", "*")

NB: it is not required that you use the conventional symbol for the
operation, but the symbol must appear in the alphabet of symbols from
the representation object.

TODO: pass in a custom representation object (and allow that object to
have a custom alphabet set by callers)

TODO: check that the operator symbol appears in Representation.alphabet
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout
from keras.models import load_model
from keras.callbacks import TensorBoard as TensorBoardCallback
from keras.callbacks import EarlyStopping as EarlyStoppingCallback
from keras.callbacks import ReduceLROnPlateau as ReduceLROnPlateauCallback
import tensorflow as tf

from time import strftime as time_to_string, gmtime as current_time
from os.path import join

from .data import Data
from .representation import Representation

class Arithmodel ():
  def __init__ (self, name, operator_symbol):
    self.model_name = name
    self.operator_symbol = operator_symbol
    self.representation = Representation ()
    self.timestamp = time_to_string ("%Y-%m-%d-%H-%M", current_time ())
    self.training_data_size = 100000
    
  def __create_model (self, input_shape):
    """create a neural network to do the lernin"""
    with tf.name_scope ("RNN"):
      layers = [
      LSTM (100, input_shape=input_shape),
      RepeatVector (self.representation.num_digits),
      LSTM (50, return_sequences=True),
      TimeDistributed (Dense (len (self.representation.alphabet), activation="softmax"))]
    
      model = Sequential (layers)
      
      print (model.summary ())
      
    with tf.name_scope ("compile"):
      model.compile (optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", "categorical_crossentropy", "mean_squared_error"])
      
    return model
  
  def train (self, target_fn, num_epochs=200, batch_size=10):
    """train a model to learn the target_fn function,
    stores a tensorboard log in the logs/<model_name>/<timestamp> folder.
    saves the model to <model_name>.hdf5
    """
    # create the callbacks
    tb = TensorBoardCallback (
      log_dir= join ("logs", self.model_name, self.timestamp),
      histogram_freq=0,
      batch_size=32,
      write_graph=True,
      write_grads=False,
      write_images=False,
      embeddings_freq=0,
      embeddings_layer_names=None,
      embeddings_metadata=None)
   
    lr = ReduceLROnPlateauCallback (monitor="accuracy")
    
    # create some data
    D = Data ((0, 1000))
    Xs, Ys = D.create_data (self.training_data_size, target_fn)
  
    Xs = [self.representation.operation_to_one_hot (x, self.operator_symbol) for x in Xs]
    Ys = [self.representation.number_to_one_hot (y) for y in Ys]
    
    input_shape = (len (Xs[0]), len (self.representation.alphabet))
    print ("input shape : ", input_shape)
  
    model = self.__create_model (input_shape)
    
    # train
    with tf.name_scope ("train"):
      model.fit (Xs, Ys,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[tb, lr])
        
    model.save (self.model_name + ".hdf5")
    
    return model
  
  def predict (self):
    model = load_model (self.model_name + ".hdf5")
    Xs = [self.representation.operation_to_one_hot ((int (x), int (y)), self.operator_symbol)]
    ys = model.predict (np.array (Xs))
    print (ys)
    return self.representation.one_hot_to_number (ys)