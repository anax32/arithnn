"""Converts number to one_hot sequence representations for a neural network
to process.

Internally, the representation must have an alphabet of symbols, including operators
and space (space symbol is used for padding by the NN).
Input numbers are converted to strings, and then, via the alphabet, converted to
one_hot indices into the mapping.

e.g.
index four is the symbol '5',
one_hot representation of four is '0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0' (assuming eleven elements in the alphabet)
Therefore, the number 55 is coverted to string:
'55'
and then to symbol index:
'4,4'
and then to one_hot sequence
[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]

TODO: have the caller pass in an alphabet (or the option to do so)
TODO: vary the number of digits in a representation
"""

class Representation ():
  def __init__ (self):
    self.alphabet = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "x", "%", "/", "-", " "]
    self.mapping = dict ((s, i) for i, s in enumerate (self.alphabet))
    self.num_digits = 4     # max number of digits in string rep
  
  def num_string (self, num):
    """convert a number to a string, padding with spaces
    upto the maximum number of digits for this context"""
    return str (num).rjust (self.num_digits)
 
  def sum_to_string (self, operands, operator):
    """convert the pair of operands for a function to
    string representation using infix notation"""
    return self.num_string (operands[0]) + \
           operator + \
           self.num_string (operands[1])
           
  @staticmethod
  def to_one_hot (L, number_of_symbols):
    """transforms a list of symbols into a list of lists, where each
    new list is a one_hot mapping of a symbol in the input list"""
    return [[(i==x)&1 for i in range (number_of_symbols)] for x in L]
   
  def number_to_one_hot (self, number):
    """convert an integer to a one_hot sequence"""
    symbolic_list = self.num_string (number)
    mapped_symbols = [self.mapping[s] for s in symbolic_list]
    return Representation.to_one_hot (mapped_symbols, len (self.mapping.keys ()))
    
  def operation_to_one_hot (self, operands, operator):
    """converts a pair of operands and an operator to a one_hot
    representation using the given mapping"""
    symbolic_list = list (self.sum_to_string (operands, operator))
    mapped_symbols = [self.mapping[s] for s in symbolic_list]
    return Representation.to_one_hot (mapped_symbols, len (self.mapping.keys ()))
    
  def one_hot_to_symbols (self, one_hot):
    """converts a one_hot representation of a symbol back to
    a symbolic string."""
    mapped_symbols = [np.argmax (i) for i in one_hot[0]]
    symbolic_list = [next (key for key, value in self.mapping.items () if value == i) for i in mapped_symbols]
    return ''.join (symbolic_list)